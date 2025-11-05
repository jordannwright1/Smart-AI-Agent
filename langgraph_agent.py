import os
import re
import smtplib
import json
import time
from typing import TypedDict, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
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
    llm = ChatOllama(model="gemma3:4b", temperature=0.7)
else:
    llm = None
    st.error("LLM not available due to missing dependencies.")


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
    st.success("🧠 Memory cleared!")

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
        self.browser = self.playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    def new_page(self):
        return self.browser.new_page()
    def close(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass
        BrowserManager._instance = None

def ensure_browser():
    if "browser_manager" not in st.session_state or st.session_state["browser_manager"] is None:
        try:
            st.session_state["browser_manager"] = BrowserManager.get_instance()
        except RuntimeError as e:
            st.error(str(e))
            raise
    return st.session_state["browser_manager"]

# ============ SMART FILTERING ============
def score_job_relevance(job_data: Dict, job_title: str, location: str, exp_level: str, min_salary: Optional[int]) -> int:
    """Fast text-based scoring before expensive Playwright check. ENHANCED LOCATION FILTER."""
    title = job_data.get("title", "").lower()
    body = job_data.get("body", "").lower()
    url = job_data.get("url", "").lower()
    combined = title + " " + body
    
    initial_score = 0
    
    # 1. Role Relevance (Kept strong)
    title_words = [w for w in job_title.lower().split() if len(w) > 2]
    matched_core_words = 0
    
    for word in title_words:
        if word in title:
            initial_score += 25 
            matched_core_words += 1
        elif word in combined:
            initial_score += 5  
            matched_core_words += 1

    required_keywords = [w for w in ["ai", "developer", "engineer", "machine learning"] if w in job_title.lower()]
    present_count = sum(1 for keyword in required_keywords if keyword in combined)
    
    if present_count == 0 and required_keywords:
        return 0 # HARD FAIL - Must contain primary role keyword
    
    if matched_core_words == len(title_words) and len(title_words) > 1:
        initial_score += 40 
    
    if required_keywords and present_count < len(required_keywords):
        initial_score -= 40 * (len(required_keywords) - present_count) 
    
    # 2. Location Match (CRITICAL ENHANCEMENT)
    is_location_filtered = location and location.lower() not in ["any", "anywhere"]
    
    if is_location_filtered:
        loc_words = location.lower().replace(',', '').split()
        location_score_increase = 0
        location_found_anywhere = False
        
        for loc_word in loc_words:
            if len(loc_word) > 2:
                if loc_word in title or loc_word in url:
                    location_score_increase += 50 # Strong signal
                    location_found_anywhere = True
                elif loc_word in body:
                    location_score_increase += 5 
                    location_found_anywhere = True
        
        # --- NEW HARD-KILL RULE: ZERO TOLERANCE FOR LOCATION ABSENCE ---
        if is_location_filtered and not location_found_anywhere:
             # If the location is absolutely not mentioned in the entire snippet, hard fail.
             return 0 

        initial_score += location_score_increase
            
        # Hard check for "Remote" when location is specified (e.g., "Remote in NYC")
        if any(x in title for x in ["remote", "wfh"]) and not any(x in location.lower() for x in ["remote", "wfh"]):
            initial_score -= 15 
            
        # Strictness Check for Location Mismatch (FOREIGN LOCATIONS & COMPETING CITIES)
        competing_locs_to_penalize = ["palo alto", "san francisco", "boston", "seattle", "austin", "chicago", "miami", "toronto", "houston", "ohio"]
        foreign_locs = ["delhi", "munich", "india", "germany", "london", "europe", "asia"] 
        
        for loc in competing_locs_to_penalize + foreign_locs:
             # Penalize if a major competing location is found
             if loc in combined and loc not in location.lower():
                 if loc in foreign_locs:
                     initial_score -= 100 # ABSOLUTE HARD KILL for foreign locations
                 else:
                     initial_score -= 70 # EXTREME PENALTY for major competing locations
    
    # 3. Experience level filtering 
    if exp_level:
        exp_lower = exp_level.lower()
        if "junior" in exp_lower or "entry" in exp_lower:
            if any(x in combined for x in ["senior", "sr.", "lead", "principal", "director", "manager"]):
                initial_score -= 50 
            if any(x in combined for x in ["junior", "entry", "associate", "early career"]):
                initial_score += 8 
        elif "senior" in exp_lower:
            if any(x in combined for x in ["junior", "entry", "intern", "associate"]):
                initial_score -= 50 
            if any(x in combined for x in ["senior", "sr.", "lead", "principal"]):
                initial_score += 8 
    
    # 4. Salary indicators 
    if min_salary:
        salary_match = re.search(r'\$?(\d{2,3})k|\$(\d{3,3},\d{3})', combined)
        if salary_match:
            try:
                raw_salary = salary_match.group(1) or salary_match.group(2)
                found_salary = int(raw_salary.replace('k', '000').replace(',', ''))
                if found_salary >= min_salary:
                    initial_score += 10
            except:
                pass
    
    # 5. Negative signals 
    if any(neg in combined for neg in ["expired", "closed", "filled", "no longer accepting", "Sorry this job is no longer available. The Similar Jobs shown below might interest you.", "Sorry, this job was removed"]):
        initial_score -= 75 
    
    return max(0, initial_score)

def validate_job_playwright(job_url: str, job_title: str) -> Optional[Dict[str, str]]:
    """Quick Playwright validation for open status."""
    if not sync_playwright: return None
    try:
        browser_manager = ensure_browser()
        page = browser_manager.new_page()
        page.set_default_navigation_timeout(8000)
        page.goto(job_url, timeout=8000, wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        
        # Check for closed indicators 
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
        # Assume open on timeout or connection error to maximize recall
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

    # --- 1. CRITERIA EXTRACTION (using LLM) ---
    prompt = f"""Extract job search criteria from this query:
1. Job Title (2-4 words)
2. Location (city/state or "anywhere")
3. Experience Level ("junior", "mid", "senior", or "any")
4. Count (number of jobs requested, default 10)
5. Minimum Salary (number only, or "none")

Query: {query}

Return in format:
Job Title: [title]
Location: [location]
Experience: [level]
Count: [number]
Salary: [number or none]"""
    
    crit_text = normalize_output(llm.invoke(prompt))
    
    job_title = "Software Engineer"
    location = "anywhere"
    exp_level = "any"
    desired_count = 10
    min_salary = None
    
    for line in crit_text.splitlines():
        if "Job Title:" in line:
            extracted = line.split(":", 1)[1].strip()
            if extracted and extracted.lower() not in ["none", "any"]:
                job_title = extracted
        elif "Location:" in line:
            extracted = line.split(":", 1)[1].strip()
            if extracted:
                location = extracted
        elif "Experience:" in line:
            extracted = line.split(":", 1)[1].strip()
            if extracted and extracted.lower() != "none":
                exp_level = extracted
        elif "Count:" in line:
            try:
                extracted = line.split(":", 1)[1].strip()
                num = int(re.search(r'\d+', extracted).group())
                desired_count = max(1, min(50, num))
            except:
                pass
        elif "Salary:" in line:
            try:
                extracted = line.split(":", 1)[1].strip()
                if extracted.lower() != "none":
                    num = int(re.search(r'\d+', extracted.replace('k', '000').replace(',', '')).group())
                    min_salary = num
            except:
                pass
    
    # --- 2. QUERY BUILDING AND SEARCH SOURCES (MASSIVE EXPANSION) ---
    loc_term = "" if location.lower() in ("any", "anywhere") else f'"{location}"'
    exp_term = f'"{exp_level}"' if exp_level.lower() != "any" else ""
    
    base_query = f'"{job_title}" {exp_term} {loc_term}'.strip()
    
    sources = [
        # Primary Individual Postings (Focus on view links)
        f'{base_query} site:linkedin.com/jobs/view',
        f'{base_query} site:indeed.com/viewjob',
        f'{base_query} site:glassdoor.com/job-listing',
        f'{base_query} site:jobs.lever.co',
        f'{base_query} site:boards.greenhouse.io',
        f'{base_query} site:wellfound.com/jobs',
        f'{base_query} site:careers.google.com/jobs',
        f'{base_query} site:jobs.apple.com/en-us/job',
        f'{base_query} site:amazon.jobs/en/jobs',
        
        # Secondary Job Boards (More likely to return search pages, but sometimes direct links)
        f'{base_query} site:ziprecruiter.com',
        f'{base_query} site:builtin.com',
        f'{base_query} site:dice.com',
        f'{base_query} site:monster.com/job',
        f'{base_query} site:careerbuilder.com/job',

        # --- ADDITIONAL SOURCES (10 NEW DOMAINS) ---
        f'{base_query} site:remotely.jobs/job',
        f'{base_query} site:hired.com/job-listings',
        f'{base_query} site:flexjobs.com/job',
        f'{base_query} site:simplyhired.com/job',
        f'{base_query} site:jora.com/job',
        f'{base_query} site:careerjet.com/job',
        f'{base_query} site:adzuna.com/details/job',
        f'{base_query} site:ladders.com/job',
        f'{base_query} site:clearancejobs.com/jobs',
        f'{base_query} site:themuse.com/jobs/view',
    ]

    random.shuffle(sources)
    
    results_raw = []
    seen_urls = set()
    
    # Regex to discard known generic search result pages (CRITICAL FOR INDIVIDUAL POSTS)
    search_list_re = re.compile(
        r'ziprecruiter\.com/(Jobs/[^/]+-Jobs|jobs\?|jobs/search)|'
        r'builtin\.com/(jobs/|categories/)|'               
        r'dice\.com/jobs\?q=|'
        r'linkedin\.com/jobs/$|'
        r'indeed\.com/jobs\?q='
    )
    
    try:
        with DDGS() as ddgs:
            for search_query in sources:
                # Use a high multiplier (15x) for recall
                if len(results_raw) >= desired_count * 15: 
                    break
                try:
                    # Search deep for each source
                    for r in ddgs.text(search_query, max_results=100):
                        url = r.get("href", "")
                        
                        if not url or url in seen_urls:
                            continue
                        if any(x in url for x in ["bing.com/aclick", "doubleclick", "googleads"]):
                            continue
                        
                        url_lower = url.lower()

                        # --- HARD DISCARD GENERIC SEARCH LINKS ---
                        if search_list_re.search(url_lower):
                             continue
                        
                        # Basic URL validation: ensure it's a specific job view link, not a general search page
                        if not any(x in url_lower for x in ["/job", "/career", "/position", "/jobs/view", "/viewjob"]):
                            continue
                        
                        
                        seen_urls.add(url)
                        results_raw.append({
                            "url": url,
                            "title": r.get("title", "")[:150],
                            "body": r.get("body", "")[:300]
                        })
                        
                        if len(results_raw) >= desired_count * 15:
                            break
                except Exception as e:
                    logging.warning(f"DDGS search failed for query '{search_query[:30]}...': {e}")
                    continue
    except requests.exceptions.ConnectTimeout as e:
        return f"⚠️ Job search failed: The underlying search engine timed out during the initial query. ({e})"
    except Exception as e:
        return f"⚠️ Job search failed during DDGS operation: {e}"

    
    if not results_raw:
        return f"⚠️ No results found for '{job_title}' in '{location}' (Raw count: 0)"
    
    # --- 3. SCORE AND FILTER ---
    scored_jobs = []
    MIN_RELEVANCE_SCORE = 10 
    
    for job in results_raw:
        score = score_job_relevance(job, job_title, location, exp_level, min_salary)
        if score > MIN_RELEVANCE_SCORE: 
            scored_jobs.append((score, job))
    
    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top candidates for validation (3x desired count)
    candidates_for_validation = [job for _, job in scored_jobs[:desired_count * 3]]
    
    if not candidates_for_validation:
        return f"⚠️ No relevant jobs found after fast filtering for '{job_title}' in '{location}' (Scored count: {len(scored_jobs)})"
    
    # --- 4. PARALLEL PLAYWRIGHT VALIDATION (Live Check) ---
    validated_jobs = []
    
    try:
        ensure_browser()
    except RuntimeError:
        return f"⚠️ Browser initialization failed. Could not perform validation."

    
    def validate_job_wrapper(job_data):
        result = validate_job_playwright(job_data["url"], job_data.get("title", "Job Posting"))
        return result
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(validate_job_wrapper, job): job for job in candidates_for_validation}
        
        for future in as_completed(futures):
            if len(validated_jobs) >= desired_count: 
                [f.cancel() for f in futures if not f.done()]
                break
            result = future.result()
            if result and result.get("url"):  
                validated_jobs.append(result)
    
    # Final Truncation 
    final_jobs = validated_jobs[:desired_count]
    elapsed = time.time() - start
    
    if not final_jobs:
        return f"⚠️ No open, relevant, and location-conforming jobs found for '{job_title}' in '{location}' (Searched {len(results_raw)} raw links in {elapsed:.1f}s)"
    
    # --- 5. FORMAT OUTPUT ---
    out = [f"**Found {len(final_jobs)} of {desired_count} requested {job_title}"]
    if exp_level.lower() != "any":
        out[0] += f" ({exp_level})"
    out[0] += " jobs"
    if location.lower() not in ["any", "anywhere"]:
        out[0] += f" in {location}"
    if min_salary:
        out[0] += f" (${min_salary:,}+)"
    out[0] += f"** (took {elapsed:.1f}s)\n"
    
    for i, job in enumerate(final_jobs, 1):
        title = job.get('title', 'Job Posting')
        url = job.get('url', '')
        if title and url:
            out.append(f"{i}. **{title}** \n   {url}\n")
        else:
            continue
    
    if len(final_jobs) < desired_count:
        out.append(f"\n⚠️ Found {len(final_jobs)}/{desired_count} jobs. Searched {len(results_raw)} raw links, filtered to {len(scored_jobs)} candidates.")
    
    filters = ["✅ Role relevance", "✅ Open status check"]
    if location.lower() not in ["any", "anywhere"]:
        filters.append(f"✅ Strict Location ({location})")
    if exp_level.lower() != "any":
        filters.append(f"✅ Experience ({exp_level})")
    if min_salary:
        filters.append(f"✅ Salary (${min_salary:,}+)")
    
    out.append(f"\n**Filters Applied:** {', '.join(filters)}")
    out.append(f"**Search Strategy:** {len(sources)} sources → {len(results_raw)} raw links → {len(scored_jobs)} scored → {len(candidates_for_validation)} validated → {len(final_jobs)} final")
    
    return "\n".join(out)

# ============ WEB SEARCH ============
def web_search(query: str) -> str:
    if not DDGS: return "⚠️ ddgs package not installed."
    if not llm: return "⚠️ LLM not available."
    try:
        query_opt_prompt = f"""Convert to search query (5-8 words max):
{query}
Return ONLY the query."""
        
        optimized = normalize_output(llm.invoke(query_opt_prompt)).strip('"').strip("'")
        
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(optimized, max_results=5)]
        
        if not results:
            return f"⚠️ No results for: '{optimized}'"
        
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. **{r['title']}**\n   {r['href']}\n   {r['body']}\n")
        
        combined = "\n".join(formatted)
        
        summary_prompt = f"""Summarize these search results for: "{query}"

CRITICAL RULES:
- ONLY use information from results below
- Include inline markdown links: [text](url)
- Do NOT invent information

Results:
{combined}"""
        
        summary = llm.invoke(summary_prompt)
        return normalize_output(summary)
    except Exception as e:
        return f"⚠️ Search failed: {e}"

# ============ SCRAPE URL ============
def scrape_url(url: str) -> str:
    if not sync_playwright: return "⚠️ Playwright not available."
    if not llm: return "⚠️ LLM not available."
    try:
        browser_manager = ensure_browser()
        page = browser_manager.new_page()
        page.goto(url, timeout=15000)
        page.wait_for_timeout(2000)
        elements = page.query_selector_all("p, h1, h2, h3, li")
        text = " ".join([el.inner_text() for el in elements if el.inner_text().strip()])
        page.close()
        
        if not text.strip():
            return "⚠️ No content found"
        
        summary_prompt = f"Summarize:\n\n{text[:3000]}"
        result = llm.invoke(summary_prompt)
        return normalize_output(result)
    except Exception as e:
        return f"⚠️ Failed: {e}"

# ============ EMAIL ============
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
        # Robust tool decision parsing
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
1. Only return a tool name (e.g., "job_search") if the query explicitly and clearly asks for that action.
2. For all other queries (like general conversation or feedback), return "none".
3. Return ONLY the tool name, with no formatting, code blocks or quotes.
"""

        raw_decision = normalize_output(llm.invoke(tool_prompt))

        # --- STRONG NORMALIZATION PIPELINE ---
        decision = str(raw_decision or "").strip().lower()

        # Remove markdown code fences, backticks, JSON-style wrappers, etc.
        decision = re.sub(r"^```(?:json|text)?", "", decision)
        decision = re.sub(r"```$", "", decision)
        decision = re.sub(r"^[\{\[\(\"\']+|[\}\]\)\"\']+$", "", decision)
        decision = re.sub(r"[^a-z_]+", "", decision)

        # Now ensure it’s a known tool or “none”
        tools = ["web_search", "job_search", "scrape_url", "send_email"]
        if decision not in tools:
            decision = "none"

        tools_to_use = [decision] if decision in tools else []

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
                email_prompt = f"Write email for:\n{collected or query}\n\nInclude Subject: line"
                result = normalize_output(llm.invoke(email_prompt))
                subj, body = extract_subject_and_body(result)
                collected += f"\n\n{send_email(subj, body, recipient)}"

        # --------------------------
        # RESPONSE GENERATION
        # --------------------------
        if tools_to_use and collected.strip():
            response_prompt = f"""Respond briefly (2-3 sentences) to the user's initial request based on the tool results provided below. 
If the request was for information, use the results directly.

Tool Results:
{collected}

Only use info from above. Be conversational."""
            response = normalize_output(llm.invoke(response_prompt))
            output = f"{response}\n\n---\n{collected}"
        else:
            conv_prompt = f"""Context:\n{context}\n\nUser: {query}\n\nRespond naturally and conversationally. Do NOT mention tools."""
            output = normalize_output(llm.invoke(conv_prompt))

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

        # Enhanced cleaning for any leftover LLM artifacts
        final_output = re.sub(r"^```(?:json|text)?|```$", "", final_output, flags=re.MULTILINE)
        final_output = re.sub(r"^(\[.*?\]|\{.*?\})\s*", "", final_output, flags=re.DOTALL)
        final_output = re.sub(r"^(output|result)\s*[:\-]\s*", "", final_output, flags=re.IGNORECASE)
        final_output = final_output.strip()

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
