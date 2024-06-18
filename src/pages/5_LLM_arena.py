import collections
if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping
    collections.MutableSet = collections.abc.MutableSet

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
from openai import OpenAI, AsyncOpenAI
from data_utils import load_templates, save_template, load_to_df, get_cos_sim
from llm_utils import get_embedding_runpod, async_call_llm, RateLimiter
from config import AVAILABLE_MODELS
from streamlit_utils import show_df, templates_form, remove_duplicates
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.bottom_container import bottom
import base64
import uuid
import random
from datetime import datetime
import pickle
from datetime import datetime

def save_dataframe(filename=None):
    if filename is None:
        filename = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f'saved_labeled_data/df_{filename}.pkl'
    if "df" in st.session_state:
        with open(filename, 'wb') as f:
            pickle.dump(st.session_state.df, f)
        
        
st.set_page_config(page_title="LLM Arena", layout="wide")
st.title("LLM Arena 🏟️")

def style_page():
    st.markdown(
        """
        <style>
        .spinner {
            display: inline-block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

style_page()

if 'models' not in st.session_state:
    st.session_state.models = [model["model"] for model in AVAILABLE_MODELS]

if len(st.session_state.models) < 2:
    st.write("You need to select at least two models to compare.")
    st.stop()

if 'messages1' not in st.session_state:
    st.session_state.messages1 = []
if 'messages2' not in st.session_state:
    st.session_state.messages2 = []


files = glob.glob('saved_labeled_data/df_*.pkl')
if len(files) > 0:
    selected_file = st.selectbox('Select file to load', files)
    if selected_file:
        with open(selected_file, 'rb') as file:
            st.session_state.df = pickle.load(file)
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Question', 'Answer1', 'Answer2', 'Model1', 'Model2', 'Choice', 'Created_at', 'History'])

st.write(st.session_state.df)

with st.form("generation_params_form"):
    st.sidebar.markdown('### LLM params')
    model_1 = st.sidebar.selectbox("Select a model_1:", [model["model"] for model in AVAILABLE_MODELS])
    model_2 = st.sidebar.selectbox("Select a model_2:", [model["model"] for model in AVAILABLE_MODELS])
    
    model_creds_1 = next((model for model in AVAILABLE_MODELS if model["model"] == model_1), None)
    model_creds_2 = next((model for model in AVAILABLE_MODELS if model["model"] == model_2), None)
    
    max_tokens = st.sidebar.number_input("max_output_tokens", min_value=1, max_value=4096, value=512)
    temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=1.0, value=0.0)

selected_template, system_prompt = templates_form(default_template='saved_templates/chatting/base.txt')

filename = st.text_input('Filename')
if st.button('Save Session Data'):
    save_dataframe(filename=filename)
    st.success('Session data saved!')
    
    
col1, col2 = st.columns(2)

meta_1 = col1.empty()
meta_2 = col2.empty()

meta_1.write(f"## :blue[Model 1: {model_1}]")
meta_2.write(f"## :red[Model 2: {model_2}]")

body_1 = col1.empty()
body_2 = col2.empty()


client = AsyncOpenAI(
      base_url = "https://openrouter.ai/api/v1",
      api_key = os.getenv("OPENROUTER_API_KEY")
    )

def get_user_input():
    if 'user_input' not in st.session_state:
        st.session_state.user_input = None
    st.session_state.user_input = st.chat_input("Message both models", key="unique_input")
    return st.session_state.user_input

def add_message(role, content):
    st.session_state.messages1.append({'role': role, 'content': content})
    st.session_state.messages2.append({'role': role, 'content': content})

async def run_prompt(placeholder, model, message_history):
    with placeholder.container():
        for message in message_history:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])
        assistant = st.chat_message(name="assistant")

        with open("images/loading-gif.gif", "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        assistant.html(f"<img src='data:image/gif;base64,{data_url}' class='spinner' width='25' />")

    messages = [
        {"role": "system", "content": system_prompt},
        *message_history
    ]

    request_id = str(uuid.uuid4())
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    streamed_text = ""
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None:
            streamed_text += chunk_content
            with placeholder.container():
                for message in message_history:
                    chat_entry = st.chat_message(name=message['role'])
                    chat_entry.write(message['content'])
                assistant = st.chat_message(name="assistant")
                assistant.write(streamed_text)

    message_history.append({"role": "assistant", "content": streamed_text})

    return streamed_text

async def main():
    answer1, answer2 = await asyncio.gather(
        run_prompt(body_1, model=model_1, message_history=st.session_state.messages1),
        run_prompt(body_2, model=model_2, message_history=st.session_state.messages2)
    )
    vote()


def save_to_dataframe(question, answer1, answer2, model1, model2, choice, created_at, history):
    new_row = pd.DataFrame([{
        'Question': question,
        'Answer1': answer1,
        'Answer2': answer2,
        'Model1': model1,
        'Model2': model2,
        'Choice': choice,
        'Created_at': created_at,
        'History': str(history),
        'System_prompt': system_prompt
    }])
    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    st.write(st.session_state.df)

def do_vote(choice):
    st.session_state.choice = choice
    model_1_display = model_1.replace(":", "\\:")
    model_2_display = model_2.replace(":", "\\:")

    if choice == "model1":
        vote_choice = f":blue[{model_1_display}]"
        st.session_state.messages2 = list(st.session_state.messages1)
    elif choice == "model2":
        vote_choice = f":red[{model_2_display}]"
        st.session_state.messages1 = list(st.session_state.messages2)
    else:
        vote_choice = ":grey[Both the same]"

    st.toast(f"""##### Your choice: {vote_choice}""", icon='🗳️')

    # Save data frame after voting
    save_to_dataframe(
        st.session_state.user_input,
        st.session_state.messages1[-1]['content'],
        st.session_state.messages2[-1]['content'],
        model_1,
        model_2,
        choice,
        datetime.now(),
        st.session_state.messages1[:-1]
    )

def vote():
    with bottom():
        col1, col2, col3 = st.columns(3)
        model1 = col1.button("Model 1 👈", key="model1", on_click=do_vote, args=["model1"])
        model2 = col2.button("Model 2 👉", key="model2", on_click=do_vote, args=["model2"])
        neither = col3.button("Both the same 🤝", key="same", on_click=do_vote, args=["same"])

with bottom():
    voting_buttons = st.empty()
    prompt = get_user_input()
    new_found = st.empty()
    with new_found.container():
        if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
            with stylable_container(
                key="next_round_button",
                css_styles="""
                    button {
                        background-color: green;
                        color: white;
                        border-radius: 10px;
                        width: 100%
                    }
                    """,
            ):
                new_round = st.button("New Round", key="new_round", on_click=lambda: [st.session_state.messages1.clear(), st.session_state.messages2.clear(), st.session_state.pop('vote', None)])

# Render existing state
if "vote" in st.session_state:
    model_1_display = model_1.replace(":", "\\:")
    model_2_display = model_2.replace(":", "\\:")
    meta_1.write(f"## :blue[Model 1: {model_1_display}]")
    meta_2.write(f"## :red[Model 2: {model_2_display}]")

if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
    with body_1.container():
        for message in st.session_state.messages1:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])

    with body_2.container():
        for message in st.session_state.messages2:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])

if prompt:
    if prompt == "":
        st.warning("Please enter a prompt")
    else:
        add_message("user", prompt)
        st.session_state.messages2 = list(st.session_state.messages1)  # Ensure both histories are in sync
        asyncio.run(main())
