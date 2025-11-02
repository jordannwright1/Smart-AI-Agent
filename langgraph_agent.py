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
