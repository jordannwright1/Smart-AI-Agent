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
