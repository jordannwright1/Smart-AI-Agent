import os
import re
import smtplib
import json
import time
from typing import TypedDict, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random 

from email.mime.text import MIMEText
from dotenv import load_dotenv
import streamlit as st

# Conditional imports to handle potential local setup issues
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    st.warning("DDGS not installed. Job search will not function.")
    
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
    st.warning("ChatOllama not installed.")

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None
    st.warning("LangGraph not installed.")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    sync_playwright = None
    PlaywrightTimeout = Exception
    st.warning("Playwright not installed.")

# Setup logging
logging.basicConfig(level=logging.INFO)
# Silence Streamlit's annoying multithreading warning
logging.getLogger('streamlit').setLevel(logging.ERROR)

# ============ ENV SETUP ============
load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD")

# ============ STREAMLIT SETUP ============
st.set_page_config(page_title="Smart AI Agent", layout="wide")
st.title("🤖 Smart LangGraph AI Agent")
st.write("Enhanced AI Agent: Multi-source job search with advanced filtering, web search and scraping, and sends emails.")

# ============ LLM SETUP ============
if ChatOllama:
    # Set model to be less resource intensive if possible
    llm = ChatOllama(model="gemma3:4b", temperature=0.7)
else:
    llm = None
    st.error("LLM not available due to missing dependencies.")

# ============ AGGRESSIVE CACHING ============
# Cache for expensive LLM calls that rely only on the input query
LLM_CACHE = {} 
JOB_CRITERIA_CACHE = {}

# ============ MEMORY ============
MEMORY_FILE = "conversation_memory.json"

def load_memory() -> List[Dict[str, str]]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []

def save_memory(memory: List[Dict[str, str]]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory[-10:], f, ensure_ascii=False, indent=2)

def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    # Clear internal caches as well
    LLM_CACHE.clear()
    JOB_CRITERIA_CACHE.clear()
    st.success("🧠 Memory and caches cleared!")

# ============ UTILITIES ============
def normalize_output(result: Any) -> str:
    if hasattr(result, "content"):
        return result.content.strip()
    if isinstance(result, dict):
        for key in ["content", "output", "output_text", "message", "result"]:
            if key in result:
                return str(result[key]).strip()
    if isinstance(result, (list, tuple)):
        return "\n".join([normalize_output(r) for r in result if r])
    if isinstance(result, str):
        return result.strip()
    return str(result)

def extract_subject_and_body(text: str) -> tuple[str, str]:
    subject_match = re.search(r"Subject:\s*(.*)", text, flags=re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()
        body = re.sub(r"Subject:.*", "", text, count=1, flags=re.IGNORECASE).strip()
    else:
        lines = [l for l in text.splitlines() if l.strip()]
        subject = lines[0][:80].strip(". ") if lines else "Automated Message"
        body = text
    return subject or "Automated Message", body

# ============ PLAYWRIGHT SINGLETON ============
class BrowserManager:
    _instance = None
    def __init__(self):
        if not sync_playwright:
             raise RuntimeError("Playwright is not installed or available.")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-software-rasterizer"]) 
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    def new_page(self):
        page = self.browser.new_page()
        # Aggressive route block for common unnecessary resources (images, fonts, CSS)
        page.route("**/*", lambda route: route.abort() 
                  if route.request.resource_type in ["image", "font", "media", "stylesheet"] 
                  else route.continue_())
        return page
    def close(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass
        BrowserManager._instance = None

def ensure_browser():
    # Only create if not already in session state (persistence across runs)
    if "browser_manager" not in st.session_state or st.session_state["browser_manager"] is None:
        try:
            st.session_state["browser_manager"] = BrowserManager.get_instance()
        except RuntimeError as e:
            st.error(str(e))
            raise
    return st.session_state["browser_manager"]


# ============ SMART FILTERING ============
# Score job relevance (no changes needed, this is fast text processing)
def score_job_relevance(job_data: Dict, job_title: str, location: str, exp_level: str, min_salary: Optional[int]) -> int:
    """
    Enhanced job relevance scoring with strict location enforcement.
    Returns 0 for jobs that don't match location criteria.
    """
    title = job_data.get("title", "").lower()
    body = job_data.get("body", "").lower()
    url = job_data.get("url", "").lower()
    combined = title + " " + body + " " + url

    score = 0

    # -----------------------------
    # 1. Role relevance
    # -----------------------------
    title_words = [w for w in job_title.lower().split() if len(w) > 2]
    matched_core_words = 0
    for word in title_words:
        if word in title:
            score += 35
            matched_core_words += 1
        elif word in combined:
            score += 15
            matched_core_words += 1

    required_keywords = [w for w in ["ai", "developer", "engineer", "machine learning", "software engineer"] if w in job_title.lower()]
    present_count = sum(1 for k in required_keywords if k in combined)
    if required_keywords and present_count == 0:
        return 15
    if matched_core_words == len(title_words) and len(title_words) > 1:
        score += 40
    if required_keywords and present_count < len(required_keywords):
        score -= 40 * (len(required_keywords) - present_count)

    # -----------------------------
    # 2. Location strictness
    # -----------------------------
    is_location_filtered = location.lower() not in ["any", "anywhere"]
    if is_location_filtered:
        loc_normalized = location.lower().replace(",", "").replace(".", "").strip()
        
        # Map to state abbreviation if applicable
        STATE_ABBR = {
            "alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca",
            "colorado":"co","connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga",
            "hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks",
            "kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma",
            "michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt",
            "nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm",
            "new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
            "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
            "south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt",
            "virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy"
        }
        loc_variants = [loc_normalized]
        if loc_normalized in STATE_ABBR:
            loc_variants.append(STATE_ABBR[loc_normalized])
        
        # Check for match
        location_found = any(v in combined for v in loc_variants)

        # Hard kill if not found
        if not location_found:
            return 0

        # Penalize competing US cities/states
        US_LOCATIONS = [s for s in STATE_ABBR.keys() if s != loc_normalized]
        for us_loc in US_LOCATIONS:
            if us_loc in combined and us_loc not in loc_variants:
                score -= 50

        # Hard kill foreign locations
        FOREIGN_LOCS = ["india","delhi","munich","germany","london","europe","asia","canada"]
        if any(f in combined for f in FOREIGN_LOCS):
            return 0

        # Remote handling
        if any(x in title for x in ["remote","wfh"]) and "remote" not in loc_normalized:
            score -= 15

    # -----------------------------
    # 3. Experience level
    # -----------------------------
    if exp_level:
        exp_lower = exp_level.lower()
        if "junior" in exp_lower or "entry" in exp_lower:
            if any(x in combined for x in ["senior","sr.","lead","principal","director","manager", "Senior", "Sr.", "Lead", "Principal", "Director", "Manager"]):
                score -= 50
            if any(x in combined for x in ["junior","entry","associate","early career", "entry level", "intern"]):
                score += 10
        elif "senior" in exp_lower:
            if any(x in combined for x in ["junior","entry","intern","associate"]):
                score -= 50
            if any(x in combined for x in ["senior","sr.","lead","principal"]):
                score += 10

    # -----------------------------
    # 4. Salary indicators
    # -----------------------------
    if min_salary:
        salary_match = re.search(r'\$?(\d{2,3})k|\$(\d{3,3},\d{3})', combined)
        if salary_match:
            try:
                raw_salary = salary_match.group(1) or salary_match.group(2)
                found_salary = int(raw_salary.replace('k','000').replace(',',''))
                if found_salary >= min_salary:
                    score += 10
            except:
                pass

    # -----------------------------
    # 5. Negative signals
    # -----------------------------
    if any(neg in combined for neg in ["expired","closed","filled","no longer accepting",
                                       "sorry this job is no longer available", "is no longer available for applications"]):
        return 0

    return max(0, score)

# Playwright validation functions (no changes needed, kept optimized)
def validate_job_playwright_fast(job_url: str) -> bool:
    """Tier 1: Fast check using only title/closed selectors after aggressive resource blocking."""
    if not sync_playwright: return True 
    try:
        browser_manager = ensure_browser()
        page = browser_manager.new_page() 
        
        page.set_default_navigation_timeout(3000) 
        page.goto(job_url, timeout=3000, wait_until="domcontentloaded") 
        page.wait_for_timeout(200) 
        
        closed_selectors = [
            "span:has-text('No longer accepting applications')",
            "span:has-text('This job is closed')",
            "div:has-text('expired')",
            "[data-test-reusability='job-details-unavailable']",
            "p:has-text('This role is no longer available')",
            "h1:has-text('job not found')",
        ]
        
        for sel in closed_selectors:
            if page.locator(sel).count() > 0:
                page.close()
                return False 
        
        page.close()
        return True 
        
    except Exception:
        return True

def validate_job_playwright_full(job_url: str, job_title: str) -> Optional[Dict[str, str]]:
    """Tier 2: Full check for the very best candidates."""
    if not sync_playwright: return None
    try:
        browser_manager = ensure_browser()
        page = browser_manager.new_page()
        
        page.set_default_navigation_timeout(5000) 
        page.goto(job_url, timeout=5000, wait_until="domcontentloaded") 
        page.wait_for_timeout(500) 
        
        closed_selectors = [
            "span:has-text('No longer accepting applications')",
            "span:has-text('This job is closed')",
            "div:has-text('expired')",
            "[data-test-reusability='job-details-unavailable']",
            "p:has-text('This role is no longer available')",
            "div[class*='expired']",
            "h1:has-text('job not found')",
            "button:has-text('Sign In to Apply')" 
        ]
        
        for sel in closed_selectors:
            if page.locator(sel).count() > 0:
                page.close()
                return None
        
        # Get title
        title = page.title() or job_title
        page.close()
        
        return {"title": title[:150], "url": job_url}
        
    except Exception:
        return {"title": job_title, "url": job_url} if not isinstance(PlaywrightTimeout, Exception) else None


# ============ JOB SEARCH ============
def job_search(query: str) -> str:
    start = time.time()
    if not DDGS: 
        return "⚠️ ddgs package not installed or available."
    if not llm:
        return "⚠️ LLM not available."
    if not sync_playwright:
        return "⚠️ Playwright not available."

    # --- 1. CRITERIA EXTRACTION (LLM or CACHE) ---
    cache_key = query.lower()
    if cache_key in JOB_CRITERIA_CACHE:
        logging.info("Using cached job criteria.")
        job_title, location, exp_level, desired_count, min_salary = JOB_CRITERIA_CACHE[cache_key]
    else:
        prompt = f"""Extract job search criteria from this query. Return ONLY a single line of pipe-separated values in this exact order and format:
[Job Title]| [Location]| [Experience Level]| [Count]| [Minimum Salary]

Job Title: 2-4 words.
Location: city/state or "anywhere".
Experience Level: "junior", "mid", "senior", or "any".
Count: number, default 10.
Minimum Salary: digits only or "none".

Query: {query}"""
        
        crit_text = normalize_output(llm.invoke(prompt)).strip()
        parts = [p.strip() for p in crit_text.split('|')[:5]]
        
        job_title, location, exp_level, desired_count, min_salary = "Software Engineer", "anywhere", "any", 15, None
        
        if len(parts) >= 1 and parts[0] and parts[0].lower() not in ["none", "any"]:
            job_title = parts[0]
        if len(parts) >= 2 and parts[1]:
            location = parts[1]
        if len(parts) >= 3 and parts[2] and parts[2].lower() != "none":
            exp_level = parts[2]
        if len(parts) >= 4:
            try:
                num = int(re.search(r'\d+', parts[3]).group())
                desired_count = max(1, min(50, num))
            except:
                pass
        if len(parts) >= 5:
            try:
                if parts[4].lower() != "none":
                    num = int(re.search(r'\d+', parts[4].replace('k','000').replace(',','')).group())
                    min_salary = num
            except:
                pass
        
        JOB_CRITERIA_CACHE[cache_key] = (job_title, location, exp_level, desired_count, min_salary)

    # --- 2. QUERY BUILDING ---
    loc_term = "" if location.lower() in ("any", "anywhere") else f'"{location}"'
    exp_term = f'"{exp_level}"' if exp_level.lower() != "any" else ""
    base_query = f'"{job_title}" {exp_term} {loc_term}'.strip()
    
    variants = [
        base_query,
        f'{base_query} hiring now',
        f'{base_query} current openings',
        f'{base_query} apply now',
        f'{base_query} careers',
        f'{base_query} job listing',
    ]
    
    sites = [
        'linkedin.com/jobs/view',
        'wellfound.com/jobs',
        'dice.com/job-detail','careerbuilder.com/job','remotely.jobs/show',
        'simplyhired.com/job','careerjet.com/jobad',
        'adzuna.com/details', 'builtin.com/job', 'jobsgpt.org/show', 'glassdoor.com/job-listing', 'theladders.com/job', 'jobright.ai/jobs/info'
    ]

    sources = [f"{v} site:{s}" for v in variants for s in sites]
    random.shuffle(sources)

    # --- 3. FETCH DDG RESULTS WITH DEDUPLICATION ---
    results_raw = []
    seen_urls = set()

    def normalize_url(url: str) -> str:
        return url.split('?')[0].split('#')[0].strip().lower()

    def fetch_ddg_results(search_query):
        temp_results = []
        try:
            with DDGS() as ddgs:
                time.sleep(random.uniform(0.5,1.5))
                for r in ddgs.text(search_query, max_results=500):
                    r['source_query'] = search_query
                    temp_results.append(r)
        except Exception as e:
            print(f"DDG error for {search_query}: {e}")
        return temp_results

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_ddg_results, q) for q in sources]
        for f in as_completed(futures):
            for r in f.result():
                url = r.get("href") or r.get("url")
                if not url: 
                    continue
                url_norm = normalize_url(url)
                if url_norm in seen_urls: 
                    continue
                seen_urls.add(url_norm)

                # broad acceptance
                if not re.search(r"/(job|jobs|career|careers|position|vacancy|opportunit|apply|role|opening|work|join)", url.lower()):
                    continue

                results_raw.append({
                    "url": url,
                    "title": r.get("title",""),
                    "snippet": r.get("body",""),
                    "source_query": r.get("source_query")
                })

    if not results_raw:
        return f"⚠️ No results found for '{job_title}' in '{location}' (Raw count: 0)"

    # --- 4. SCORE AND FILTER ---
    scored_jobs = []
    MIN_RELEVANCE_SCORE = 5
    for job in results_raw:
        score = score_job_relevance(job, job_title, location, exp_level, min_salary)
        if score > MIN_RELEVANCE_SCORE:
            scored_jobs.append((score, job))
    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    
    candidates_for_validation = [job for _, job in scored_jobs[:desired_count*60]]
    if not candidates_for_validation:
        return f"⚠️ No relevant jobs found after filtering '{job_title}' in '{location}' (Scored count: {len(scored_jobs)})"

    # --- 5. FAST PLAYWRIGHT VALIDATION ---
    try:
        ensure_browser()
    except RuntimeError:
        return f"⚠️ Browser initialization failed."

    fast_validated_jobs = []
    def validate_job_wrapper_fast(job_data):
        return job_data if validate_job_playwright_fast(job_data["url"]) else None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(validate_job_wrapper_fast, job): job for job in candidates_for_validation}
        for future in as_completed(futures):
            result = future.result()
            if result:
                fast_validated_jobs.append(result)
                if len(fast_validated_jobs) >= desired_count*4:
                    [f.cancel() for f in futures if not f.done()]
                    break

    # --- 6. FULL VALIDATION ---
    final_candidates = fast_validated_jobs[:desired_count+55]
    validated_jobs = []
    def validate_job_wrapper_full(job_data):
        return validate_job_playwright_full(job_data["url"], job_data.get("title","Job Posting"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(validate_job_wrapper_full, job): job for job in final_candidates}
        for future in as_completed(futures):
            if len(validated_jobs) >= desired_count:
                [f.cancel() for f in futures if not f.done()]
                break
            result = future.result()
            if result and result.get("url"):
                validated_jobs.append(result)

    # --- 7. FINAL DEDUPLICATION ---
    final_jobs = []
    seen_final_urls = set()
    for job in validated_jobs:
        url_norm = normalize_url(job.get("url",""))
        if url_norm in seen_final_urls:
            continue
        seen_final_urls.add(url_norm)
        final_jobs.append(job)
    final_jobs = final_jobs[:desired_count]

    elapsed = time.time() - start
    if not final_jobs:
        return f"⚠️ No open, relevant jobs found for '{job_title}' in '{location}' (Searched {len(results_raw)} raw links in {elapsed:.1f}s)"

    # --- 8. FORMAT OUTPUT ---
    out = [f"**Found {len(final_jobs)} of {desired_count} requested {job_title}"]
    if exp_level.lower() != "any":
        out[0] += f" ({exp_level})"
    out[0] += " jobs"
    if location.lower() not in ["any","anywhere"]:
        out[0] += f" in {location}"
    if min_salary:
        out[0] += f" (${min_salary:,}+)"
    out[0] += f"** (took {elapsed:.1f}s)\n"

    for i, job in enumerate(final_jobs,1):
        title = job.get("title","Job Posting")
        url = job.get("url","")
        if title and url:
            out.append(f"{i}. **{title}** \n   {url}\n")

    if len(final_jobs) < desired_count:
        out.append(f"\n⚠️ Found {len(final_jobs)}/{desired_count} jobs. Searched {len(results_raw)} raw links, filtered to {len(scored_jobs)} candidates.")

    filters = ["✅ Role relevance","✅ Two-Tier Open status check"]
    if location.lower() not in ["any","anywhere"]:
        filters.append(f"✅ Strict Location ({location})")
    if exp_level.lower() != "any":
        filters.append(f"✅ Experience ({exp_level})")
    if min_salary:
        filters.append(f"✅ Salary (${min_salary:,}+)")
    out.append(f"\n**Filters Applied:** {', '.join(filters)}")
    out.append(f"**Search Strategy:** {len(sources)} sources → {len(results_raw)} raw links → {len(scored_jobs)} scored → {len(candidates_for_validation)} fast-validated → {len(final_candidates)} fully validated → {len(final_jobs)} final")

    return "\n".join(out)

# ============ WEB SEARCH ============
# ... (Remains the same as no significant speedup can be achieved without caching LLM response) ...
def web_search(query: str) -> str:
    if not DDGS: return "⚠️ ddgs package not installed."
    if not llm: return "⚠️ LLM not available."
    try:
        # Use cache for optimized query
        cache_key = f"web_query:{query.lower()}"
        if cache_key in LLM_CACHE:
            optimized = LLM_CACHE[cache_key]
        else:
            query_opt_prompt = f"""Convert to search query (5-8 words max):
{query}
Return ONLY the query."""
            optimized = normalize_output(llm.invoke(query_opt_prompt)).strip('"').strip("'")
            LLM_CACHE[cache_key] = optimized
        
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(optimized, max_results=5)]
        
        if not results:
            return f"⚠️ No results for: '{optimized}'"
        
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[[Result {i}]] Title: {r['title']} URL: {r['href']} Snippet: {r['body']}\n")
        
        combined = "\n".join(formatted)
        
        # Use cache for summary
        summary_cache_key = f"web_summary:{combined}"
        if summary_cache_key in LLM_CACHE:
            summary = LLM_CACHE[summary_cache_key]
        else:
            summary_prompt = f"""Summarize these search results for the user's query: "{query}"

CRITICAL RULES:
1. Provide a **natural, conversational summary** in 2-4 paragraphs.
2. Integrate **inline markdown links** using the `[text](url)` format. 
3. Only use information and URLs from the results below.

Results:
{combined}"""
            summary = normalize_output(llm.invoke(summary_prompt))
            LLM_CACHE[summary_cache_key] = summary

        return summary
    except Exception as e:
        return f"⚠️ Search failed: {e}"

# ============ SCRAPE URL ============
def scrape_url(url: str) -> str:
    if not sync_playwright: return "⚠️ Playwright not available."
    if not llm: return "⚠️ LLM not available."
    
    cache_key = f"scrape_summary:{url}"
    if cache_key in LLM_CACHE:
        logging.info(f"Using cached scrape summary for {url}")
        return LLM_CACHE[cache_key]

    try:
        browser_manager = ensure_browser()
        page = browser_manager.new_page()
        
        page.goto(url, timeout=10000) 
        page.wait_for_timeout(1000) 
        
        elements = page.query_selector_all("p, h1, h2, h3, li, blockquote, article")
        text = " ".join([el.inner_text() for el in elements if el.inner_text().strip()])
        page.close()
        
        if not text.strip():
            return "⚠️ No content found"
        
        summary_prompt = f"""Summarize the following text from the URL: {url}
The summary should be robust, detailed, and organized into brief sections (e.g., using bullet points or sub-headings) covering the main points of the article. 

Text to Summarize:
{text[:4000]}""" 
        
        result = llm.invoke(summary_prompt)
        summary = normalize_output(result)
        LLM_CACHE[cache_key] = summary # Cache the result
        return summary

    except Exception as e:
        return f"⚠️ Failed: {e}"

# ============ EMAIL ============
# ... (Remains the same) ...
def send_email(subject: str, body: str, recipient_email: str) -> str:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "❌ Failed: SENDER_EMAIL or SENDER_APP_PASSWORD not configured."
        
    body = body.replace("\\n", "\n").strip()
    msg = MIMEText("<br>".join(body.split("\n")), "html")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return f"✅ Email sent to {recipient_email}!"
    except Exception as e:
        return f"❌ Failed: {e}"

# ============ LANGGRAPH ============
class AgentState(TypedDict):
    messages: List[dict]
    input: str
    output: str

if StateGraph:
  def main_node(state: AgentState):
        if not llm:
            return {"messages": state["messages"], "input": state["input"], "output": "LLM not initialized."}

        query = state["input"].strip()
        messages = load_memory()
        messages.append({"role": "user", "content": query})
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-5:]])

        # --------------------------
        # Aggressively Optimized Tool Decision
        # --------------------------
        tool_prompt = f"""Decide which single tool to use for this query.

Available tools:
- "web_search"
- "job_search"
- "scrape_url"
- "send_email"
- "none" (for conversational response or feedback)

Query: {query}

CRITICAL RULES:
1. Return ONLY one of the tool names or "none".
2. Do not include any other text, quotes, or formatting.
3. If the query is complex or ambiguous, default to "none".
4. If the query asks to search for jobs, use "job_search".
5. If the query asks to search the internet/web/google, use "web_search".
6. If the query contains a URL and asks for a summary or review, use "scrape_url".
7. If the query contains an email address and asks to send a message, use "send_email".
8. If none of the above, use "none".

Return value:"""

        # Use cache for tool decision
        tool_cache_key = f"tool_decision:{query.lower()}"
        if tool_cache_key in LLM_CACHE:
            decision = LLM_CACHE[tool_cache_key]
            logging.info(f"Using cached tool decision: {decision}")
        else:
            raw_decision = normalize_output(llm.invoke(tool_prompt))
            decision = str(raw_decision or "").strip().lower()
            decision = re.sub(r"[^a-z_]+", "", decision)
            
            tools = ["web_search", "job_search", "scrape_url", "send_email"]
            if decision not in tools:
                decision = "none"
            
            LLM_CACHE[tool_cache_key] = decision

        tools_to_use = [decision] if decision != "none" else []

        # Email and URL extraction
        urls = re.findall(r"https?://[^\s]+", query)
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", query)
        recipient = email_match.group(0) if email_match else None

        collected = ""

        # --------------------------
        # TOOL EXECUTION
        # --------------------------
        for tool in tools_to_use:
            if tool == "web_search":
                collected += f"\n\n🔍 {web_search(query)}"
            elif tool == "job_search":
                collected += f"\n\n💼 {job_search(query)}"
            elif tool == "scrape_url" and urls:
                collected += f"\n\n🌐 {scrape_url(urls[0])}"
            elif tool == "send_email" and recipient:
                # Use cache for email writing
                email_prompt_key = f"email_draft:{collected or query}"
                if email_prompt_key in LLM_CACHE:
                    result = LLM_CACHE[email_prompt_key]
                else:
                    email_prompt = f"Write email for:\n{collected or query}\n\nInclude Subject: line"
                    result = normalize_output(llm.invoke(email_prompt))
                    LLM_CACHE[email_prompt_key] = result
                    
                subj, body = extract_subject_and_body(result)
                collected += f"\n\n{send_email(subj, body, recipient)}"

        # --------------------------
        # RESPONSE GENERATION
        # --------------------------
        if tools_to_use and collected.strip():
            # Use cache for final response summary
            response_cache_key = f"final_response:{query}:{collected}"
            if response_cache_key in LLM_CACHE:
                response = LLM_CACHE[response_cache_key]
            else:
                response_prompt = f"""Based on the tool results provided below, provide a comprehensive, conversational response to the user's initial request.

CRITICAL RULES:
1. The response should be a well-structured narrative, not just 2-3 sentences.
2. If the tool output contains links or organized information, seamlessly integrate that information into your natural language response.
3. Return ONLY a human-readable narrative summary.
Do NOT include arrays, JSON, tables, or enumerated boolean values.
Tool Results:
{collected}"""
                response = normalize_output(llm.invoke(response_prompt))
                LLM_CACHE[response_cache_key] = response
            
            output = f"{response}\n\n---\n{collected}"
        else:
            # Use cache for conversational response
            conv_cache_key = f"conv_response:{context}"
            if conv_cache_key in LLM_CACHE:
                output = LLM_CACHE[conv_cache_key]
            else:
                conv_prompt = f"""Context:\n{context}\n\nUser: {query}\n\nRespond naturally and conversationally. Do NOT mention tools."""
                output = normalize_output(llm.invoke(conv_prompt))
                LLM_CACHE[conv_cache_key] = output


        messages.append({"role": "assistant", "content": output})
        save_memory(messages)
        return {"messages": messages, "input": query, "output": output}

  def end_node(state: AgentState):
        return state

  graph = StateGraph(AgentState)
  graph.add_node("main", main_node)
  graph.add_node("end", end_node)
  graph.set_entry_point("main")
  graph.set_finish_point("end")
  graph.add_edge("main", "end")
  app = graph.compile()
else:
    app = None
    st.error("LangGraph functionality is disabled due to missing dependencies.")


# --------------------------
# STREAMLIT OUTPUT CLEANUP
# --------------------------
query = st.text_area("Enter your request:", height=120)
col1, col2, col3 = st.columns(3)
run_button_clicked = col1.button("Run Agent")

if run_button_clicked and app:
    with st.spinner("Thinking..."):
        result = app.invoke({"input": query, "messages": [], "output": ""})
        final_output = result["output"].strip()
        

        # Enhanced cleaning for any leftover LLM artifacts (Bracketed list, code fences, etc.)
        final_output = re.sub(r"^```(?:json|text)?|```$", "", final_output, flags=re.MULTILINE).strip()
        final_output = re.sub(r"^\s*\[.*?\]\s*", "", final_output, count=1, flags=re.MULTILINE | re.DOTALL).strip()
        final_output = re.sub(r"^\s*(Output|Tool:|Action:|Result|Final Answer)\s*[:\-]*\s*", "", final_output, count=1, flags=re.IGNORECASE).strip()
        final_output = re.sub(r"^\[\d+:(true|false)(,\d+:(true|false))*\]\s*", "", final_output).strip()
        st.markdown(final_output)

elif run_button_clicked and not app:
    st.error("Agent cannot run. Please check the setup warnings above.")

if col2.button("Clear Memory"):
    clear_memory()

if col3.button("Close Browser"):
    if st.session_state.get("browser_manager"):
        try:
            st.session_state["browser_manager"].close()
            st.session_state["browser_manager"] = None
            st.success("Browser closed.")
        except Exception:
            st.warning("Browser was already closed or failed to close gracefully.")
            st.session_state["browser_manager"] = None
    else:
        st.info("Browser is not currently running.")
