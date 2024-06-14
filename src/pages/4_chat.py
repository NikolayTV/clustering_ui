import re
import os
import glob
import json
from textwrap import dedent

import asyncio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

from data_utils import load_templates, save_template, load_to_df, get_cos_sim
from llm_utils import get_embedding_runpod, async_call_llm, RateLimiter
from config import AVAILABLE_MODELS
from streamlit_utils import show_df, templates_form, remove_duplicates

st.set_page_config(
    page_title="Chat with LLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Chat with LLM")

with st.form("generation_params_form"):
    st.sidebar.markdown('### LLM params')
    model_name = st.sidebar.selectbox("Select a model:", [model["model"] for model in AVAILABLE_MODELS])
    model_creds = next((model for model in AVAILABLE_MODELS if model["model"] == model_name), None)
    max_tokens = st.sidebar.number_input("max_output_tokens", min_value=1, max_value=4096, value=512)
    temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=1.0, value=0.1)

# if "system_prompt" not in st.session_state:
selected_template, system_prompt = templates_form(default_template='saved_templates/chatting/base.txt')
    # st.session_state.system_prompt = system_prompt
    
messages = [{'role': 'system', 'content': system_prompt}]

if 'messages' not in st.session_state:
    st.session_state.messages = messages

def get_user_input():
    user_input = st.text_input("user:", key="user_input")
    return user_input

def add_message(role, content):
    st.session_state.messages.append({'role': role, 'content': content})

user_input = get_user_input()

async def send_message():
    if user_input:
        add_message("user", user_input)
        with st.spinner("Generating response..."):
            response = await async_call_llm(st.session_state.messages, model_creds, max_tokens, temperature, rate_limiter=RateLimiter(50, 5))
            add_message("assistant", response['text_response'])

if st.button("Send"):
    asyncio.run(send_message())

for message in st.session_state.messages:
    if message['role'] != 'system':
        st.write(f"**{message['role'].capitalize()}:** {message['content']}")

if st.button("Clear Chat"):
    st.session_state.messages = [{'role': 'system', 'content': system_prompt}]
    