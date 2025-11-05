from langchain.tools import Tool
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent
import smtplib
from email.mime.text import MIMEText
import json
from dotenv import load_dotenv
import os
import streamlit as st
import re
from langchain.memory import ConversationBufferWindowMemory


from langchain.schema import AIMessage

def normalize_agent_output(result):
    """Ensure agent output is always returned as a plain string."""
    def extract_text(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, AIMessage):
            return (value.content or "").strip()
        if isinstance(value, (list, tuple)):
            parts = [extract_text(v) for v in value if v]
            return "\n".join([p for p in parts if p])
        if isinstance(value, dict):
            for k in ("output", "output_text", "content", "message", "result"):
                if k in value and isinstance(value[k], str):
                    return value[k].strip()
            # Recursively search deeper
            for v in value.values():
                txt = extract_text(v)
                if txt:
                    return txt
        return str(value).strip()

    try:
        return extract_text(result)
    except Exception as e:
        return f"[Error normalizing agent output: {e}]"




# Load environment variables
load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD")

# Initialize LLM
if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model="gemma3:4b", temperature=0.7)
llm = st.session_state.llm

# --- Tools ---
def web_search(query):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    return "\n\n".join([f"{r['title']} - {r['href']}\n{r['body']}" for r in results])

def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text[:2000]
    except Exception as e:
        return f"Failed to scrape: {e}"

def compose_email(*args, **kwargs):
    if args and isinstance(args[0], str):
        body = args[0]
        subject = "No Subject Provided"
    else:
        subject = kwargs.get("Subject") or kwargs.get("subject") or "No Subject Provided"
        body = kwargs.get("Body") or kwargs.get("body") or "No body provided."
    
    prompt = f"""
    You are an assistant drafting a professional but friendly email.
    Subject: {subject}

    Based on this information, compose the email body:

    {body}
    """
    return llm.invoke(prompt)

def send_email(subject, body, recipient_email, sender_email, sender_password):
    html_body = "<br>".join(body.split("\n"))
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)
    
    return f"✅ Email sent to {recipient_email} successfully!"

def send_email_tool(input_data=None, subject=None, body=None, recipient_email=None,
                    sender_email=None, sender_password=None):

    # --- Parse input ---
    if isinstance(input_data, dict):
        data = input_data
    elif isinstance(input_data, str):
        try:
            data = json.loads(input_data.replace("'", '"'))
        except Exception:
            data = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", input_data))
    else:
        data = {}

    # Merge all sources
    subject = subject or data.get("subject") or "No Subject Provided"
    body = body or data.get("body") 
    recipient = recipient_email or data.get("recipient_email")
    sender_email = sender_email or SENDER_EMAIL
    sender_password = sender_password or SENDER_PASSWORD

    if not recipient:
        return "❌ Recipient email not provided in the input."

    # --- Send the email ---
    # --- Clean up LLM-generated newlines and format for HTML email ---
    # Convert escaped newlines to real newlines first
    body = body.replace("\\n", "\n").strip()

    # Convert markdown-style bullet points into HTML list items
    body = re.sub(r"\n\s*\*\s+", "\n<li>", body)
    if "<li>" in body:
        body = "<ul>" + body + "</ul>"

    # Convert multiple newlines to paragraph breaks
    body = re.sub(r"\n{2,}", "</p><p>", body)

    # Wrap everything in paragraph tags if not already HTML
    if not body.strip().startswith("<"):
      body = f"<p>{body}</p>"

    html_body = body
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "✅ Email sent successfully!"
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"
    
def general_response_tool(input_text: str):
    """General-purpose tool to respond to user queries or summarize text."""
    prompt = f"Please respond or summarize this request clearly:\n\n{input_text}"
    return llm.invoke(prompt)


# --- Tools list for LangChain agent ---
tools = [
    Tool(
        name="WebSearch",
        func=web_search,
        description="Search the web for recent information on a topic."
    ),
    Tool(
        name="WebScraper",
        func=scrape_url,
        description="Scrape the content of a given webpage URL."
    ),
    Tool(
        name="EmailComposer",
        func=compose_email,
        description="Compose a professional email draft given a subject and summary."
    ),
    Tool(
        name="SendEmailTool",
        func=send_email_tool,
        description="Sends an email using Gmail credentials. Input JSON must include 'subject', 'body', and 'recipient_email'.",
        return_direct=True
    ),
    Tool(
        name="GeneralResponse",
        func=general_response_tool,
        description="Handles general queries, summarizes information, or returns text responses to the user.",
        return_direct=True
    ),
]

# --- Initialize memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,  # only remember last 5 exchanges
        return_messages=True
    )

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        verbose=False,
        handle_parsing_errors=True,
        memory=st.session_state.memory  # attach memory
    )


agent = st.session_state.agent


# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Smart AI Agent", layout="wide")
st.title("🤖 Smart AI Agent Dashboard")
st.write("This agent can search the web, summarize, and send emails automatically!")

query = st.text_area(
    "Enter your request (include the recipient email in the text, e.g. Send this to test@example.com):"
)

if st.button("Run Agent"):
    with st.spinner("Thinking..."):
        try:
            def extract_email(text):
                match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
                return match.group(0) if match else None

            recipient_email = extract_email(query)

            if recipient_email:
                injected_query = (
                    query
                    + f"\n\n[Note for agent: The recipient email is {recipient_email}. "
                      "If you use the SendEmailTool, include this exact address in the input JSON under 'recipient_email'.]"
                )
            else:
                injected_query = query

            past = st.session_state.memory.load_memory_variables({})

            # ✅ Call the actual agent
            raw_result = agent.invoke({"input": injected_query})
            agent_result = normalize_agent_output(raw_result)

            # ✅ Save conversation
            st.session_state.memory.chat_memory.add_user_message(query)
            st.session_state.memory.chat_memory.add_ai_message(agent_result)

            # ✅ Display result
            if "✅ Email sent" in agent_result:
                st.success(agent_result)
            else:
                st.markdown("### 🧠 Agent Output")
                st.write(agent_result)

        except Exception as e:
            st.error(f"Error: {e}")
