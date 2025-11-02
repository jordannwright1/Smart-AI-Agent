import os
import re
import smtplib
import json
from email.mime.text import MIMEText
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# =====================
# ENV SETUP
# =====================
load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD")

# =====================
# STREAMLIT CONFIG
# =====================
st.set_page_config(page_title="Smart AI Agent", layout="wide")
st.title("🤖 Smart AI Agent")
st.write("An autonomous Ollama-powered agent that can search, scrape, summarize, and send emails.")

# =====================
# LLM SETUP
# =====================
llm = ChatOllama(model="gemma3:4b", temperature=0.7)

# =====================
# MEMORY SETUP
# =====================
MEMORY_FILE = "conversation_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
            except json.JSONDecodeError:
                return []
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory[-10:], f, ensure_ascii=False, indent=2)

def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        st.success("🧠 Memory cleared!")

# =====================
# UTILITIES
# =====================
def normalize_output(result):
    """Normalize model output."""
    if hasattr(result, "content"):
        return result.content.strip()
    elif isinstance(result, dict):
        for key in ["content", "output", "output_text", "message", "result"]:
            if key in result and isinstance(result[key], str):
                return result[key].strip()
    elif isinstance(result, list):
        return "\n".join(normalize_output(r) for r in result if r)
    elif isinstance(result, str):
        return result.strip()
    return str(result)

def extract_subject_and_body(text: str):
    """Extract a subject line and email body."""
    subject_match = re.search(r"Subject:\s*(.*)", text)
    if subject_match:
        subject = subject_match.group(1).strip()
        body = re.sub(r"Subject:.*", "", text, count=1).strip()
    else:
        subject = text.split("\n")[0][:80].strip(". ")
        body = text
    return subject or "Automated Message", body

# =====================
# TOOLS
# =====================
def web_search(query: str):
    """Perform a DuckDuckGo search and return results with links."""
    from ddgs import DDGS
    try:
        # First, optimize the search query
        query_optimization_prompt = f"""Convert this user query into an effective web search query (5-8 words max).
Extract the core search terms and remove conversational elements.

User query: {query}

Return ONLY the optimized search query, nothing else."""
        
        optimized_query_result = llm.invoke(query_optimization_prompt)
        optimized_query = normalize_output(optimized_query_result).strip('"').strip("'")
        
        # Perform the search with optimized query
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(optimized_query, max_results=5)]
        
        if not results:
            return f"⚠️ No results found for search: '{optimized_query}'"
        
        # Format results with links for LLM processing
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(f"{i}. **{r['title']}**\n   Link: {r['href']}\n   Description: {r['body']}\n")
        
        # Combine for LLM
        combined = "\n".join(formatted_results)
        
        prompt = f"""Analyze these search results and provide a natural, conversational summary for this original query: "{query}"

Search was performed using: "{optimized_query}"

CRITICAL RULES:
- ONLY use information that appears in the search results below
- If you mention ANY specific fact, company, article, or resource, it MUST be from these results
- Include relevant links from the results
- If the results don't contain information relevant to the original query, say so clearly and briefly describe what was found instead
- Do NOT invent, assume, or add any external knowledge

Search results:
{combined}

Provide a helpful summary with relevant links."""
        
        summary = llm.invoke(prompt)
        return normalize_output(summary)
    except Exception as e:
        return f"⚠️ Search failed: {e}"

def scrape_url(url: str):
    """Scrape and summarize a web page."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_timeout(2000)
            elements = page.query_selector_all("p, h1, h2, h3, li")
            text = " ".join([el.inner_text() for el in elements if el.inner_text().strip()])
            browser.close()
        if not text.strip():
            return "⚠️ No visible text found or requires login."
        prompt = f"Summarize the following webpage text:\n\n{text[:3000]}"
        result = llm.invoke(prompt)
        return normalize_output(result)
    except PlaywrightTimeout:
        return "⚠️ Timeout while loading page."
    except Exception as e:
        return f"⚠️ Failed to scrape page: {e}"

def send_email(subject, body, recipient_email):
    """Send an email with a dynamic subject."""
    body = body.replace("\\n", "\n").strip()
    html_body = "<br>".join(body.split("\n"))
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return f"✅ Email sent successfully to {recipient_email}!\n**Subject:** {subject}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

# =====================
# LANGGRAPH STATE
# =====================
class AgentState(TypedDict):
    messages: List[dict]
    input: str
    output: str

# =====================
# MAIN NODE
# =====================
def main_node(state: AgentState):
    query = state["input"].strip()
    messages = load_memory()
    output_text = ""

    # Save user query
    messages.append({"role": "user", "content": query})

    # Build conversation context for tool decision
    context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-5:]])

    # Tool awareness section
    tool_prompt = f"""
You are a reasoning engine that decides which tool to use.
The available tools are:
- "web_search": for searching the internet (use ONLY when user explicitly asks to search, look up, or find information online).
- "scrape_url": for extracting and summarizing webpage content (use ONLY when a URL is provided).
- "send_email": for composing and sending emails (use ONLY when user asks to send an email and provides a recipient).

Here's the recent conversation context:
{context}

Determine which tools (if any) are needed for this user query.
IMPORTANT: Return [] (empty list) for:
- General conversation
- Memory tasks (remembering information)
- Questions that can be answered from conversation history
- Simple requests that don't require external data

Return a JSON list of tools to use (in order), e.g. ["web_search"] or ["scrape_url", "send_email"] or [].

Current user query: {query}
"""
    decision_raw = llm.invoke(tool_prompt)
    decision_text = normalize_output(decision_raw)

    try:
        # Try to find JSON array in the response
        json_match = re.findall(r"\[.*?\]", decision_text)
        if json_match:
            tools_to_use = json.loads(json_match[0])
        else:
            # If no JSON found, default to empty list (no tools)
            tools_to_use = []
    except Exception:
        # On any error, default to no tools
        tools_to_use = []

    # Identify email recipient and URLs
    urls = re.findall(r"https?://[^\s]+", query)
    match_email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", query)
    recipient_email = match_email.group(0) if match_email else None

    # Initialize collected content
    collected_info = ""

    # Execute tool chain
    for tool in tools_to_use:
        if tool == "web_search":
            # Use the original user query for search, not the processed query
            search_summary = web_search(query)
            collected_info += f"\n\n🔍 Search Results:\n{search_summary}"

        elif tool == "scrape_url" and urls:
            scrape_summary = scrape_url(urls[0])
            collected_info += f"\n\n🌐 Scraped Content:\n{scrape_summary}"

        elif tool == "send_email" and recipient_email:
            email_prompt = f"""
Recent conversation context:
{context}

Write one clear, polished email summarizing this information:
{collected_info or query}

Include both a subject line and body.
"""
            result = llm.invoke(email_prompt)
            ai_output = normalize_output(result)
            subject, body = extract_subject_and_body(ai_output)
            email_status = send_email(subject, body, recipient_email)
            collected_info += f"\n\n{email_status}"

    # If tools were used, generate a contextual response
    if tools_to_use and collected_info:
        response_prompt = f"""
You are a helpful assistant responding to a user query.

Recent conversation context:
{context}

You searched the web and found this information:
{collected_info}

Instructions:
1. Provide a natural, conversational response that addresses the user's query
2. You may interpret and summarize the search results naturally
3. CRITICAL: Only reference information that appears in the search results above
4. Do NOT add facts, companies, articles, or resources that aren't in the search results
5. If the search results don't fully answer the query, acknowledge this
6. Keep your response helpful and engaging

Respond naturally while staying grounded in the search results.
"""
        result = llm.invoke(response_prompt)
        contextual_response = normalize_output(result)
        output_text = f"{contextual_response}\n\n---\n\n**Search Results:**\n{collected_info}"

    # Fallback: no tools triggered → normal conversation
    if not tools_to_use:
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-5:]])
        prompt = f"""
You are a memory-aware assistant.
Here's the recent conversation:
{context}

User just said: "{query}"

Respond naturally.
"""
        result = llm.invoke(prompt)
        collected_info = normalize_output(result)
        output_text = collected_info
    
    # Use output_text if it wasn't set by tool execution
    if not output_text:
        output_text = collected_info
    messages.append({"role": "assistant", "content": output_text})
    save_memory(messages)

    return {"messages": messages, "output": output_text, "input": query}

# =====================
# END NODE
# =====================
def end_node(state: AgentState):
    return state
