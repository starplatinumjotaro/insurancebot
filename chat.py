import os
from openai import OpenAI
import streamlit as st
from app.utils import generate_queries_chatgpt, get_references, rag_fusion, generate_answers
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.chat_history import InMemoryHistory

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

history = InMemoryHistory()

def handle_chat():
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    if len(st.session_state['requests']) == 0 and len(st.session_state['responses']) == 0:
        st.session_state['responses'].append("How can I assist you with your insurance queries?")

    user_input = st.chat_input("Clear your doubts!!!...", key="input")

    if user_input:
        st.session_state['requests'].append(user_input)
        query_list = generate_queries_chatgpt(user_input)
        reference_list = [get_references(q) for q in query_list]
        r = rag_fusion(reference_list)
        ranked_reference_list = [doc_str for doc_str, score in r]
        ans = generate_answers(user_input, ranked_reference_list)
        st.session_state['responses'].append(ans)
        st.session_state['last_input'] = ''

    for i in range(len(st.session_state['responses'])):
        st.write(f"Bot: {st.session_state['responses'][i]}")
        if i < len(st.session_state['requests']):
            st.write(f"You: {st.session_state['requests'][i]}")