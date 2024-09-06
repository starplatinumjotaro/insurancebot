import os
from dotenv import load_dotenv
import streamlit as st
from app.chat import handle_chat
from app.utils import load_documents

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="RAG Q&A BOT", page_icon="🤖", layout="wide")

# Title of the app
st.title("INSURANCE BOT")

# Load documents
documents_path = "documents/"
load_documents(documents_path)

# Handle user input and chat interactions
handle_chat()