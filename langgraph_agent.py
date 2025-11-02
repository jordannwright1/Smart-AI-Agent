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
