import re, os
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

# File Upload Section
st.set_page_config(
    page_title="Semantic search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("Тематическое моделирование чатов")

# ROWS_LIMIT = st.sidebar.text_input("Максимальное количество строк для отображения", value=100)
# try:
#     ROWS_LIMIT = int(ROWS_LIMIT)
# except ValueError:
#     st.error("Пожалуйста, введите числовое значение для максимального количества строк")
#     ROWS_LIMIT = 100
ROWS_LIMIT = 100

# Function to add a new semantic condition
def add_condition():
    st.session_state.semantic_queries_unfilled.append('')
    st.session_state.semantic_queries.append(('', ''))
    
def remove_condition(index):
    if len(st.session_state.semantic_queries) > 1:
        st.session_state.semantic_queries_unfilled.pop(index)
        st.session_state.semantic_queries.pop(index)

# Function to update condition value in session state
def update_condition(index, value):
    st.session_state.semantic_queries[index] = value


@st.cache_data
def apply_fts(df, white_list_and, white_list_or, black_list):
    if white_list_and:
        white_terms_and = white_list_and.split(';')
        for term in white_terms_and:
            if len(term) > 0:
                df = df[df['question'].str.contains(term, case=False, na=False)]

    if white_list_or:
        white_terms_or = white_list_or.split(';')
        white_terms_or = [x for x in white_terms_or if len(x) > 0]
        df = df[df['question'].str.contains('|'.join(white_terms_or), case=False, na=False)]
    
    if black_list:
        black_terms = black_list.split(';')
        for term in black_terms:
            if len(term) > 0:
                df = df[~df['question'].str.contains(term, case=False, na=False)]
    return df

@st.experimental_fragment
def show_data_toggle(data, rows_limit):
    show_original = st.toggle(label='Show data sample', key='toggle_data')
    if show_original:
        st.dataframe(data[:100])

@st.experimental_fragment
def show_histogram_toggle():
    show = st.toggle(label='Similarity distribution', key='toggle_hist')
    if show:
        try:
            plot_histograms(cos_sim_dict)
        except Exception as e:
            print(f"Error: {e}")
        
@st.cache_data
def plot_histograms(cos_sim_dict):
    num_plots = len(cos_sim_dict)
    
    # Combine all data to get a common x-axis range
    all_data = np.concatenate(list(cos_sim_dict.values()))
    common_bins = np.linspace(all_data.min(), all_data.max(), 50)

    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 1.5 * num_plots), sharex=True, sharey=True, tight_layout=True)

    if num_plots == 1:
        axs = [axs]  # Ensure axs is always a list

    for i, (title, data) in enumerate(cos_sim_dict.items()):
        axs[i].hist(data, bins=common_bins, color='blue', alpha=0.7)
        axs[i].set_title(f'{title}')
        axs[i].set_ylabel('Frequency')
    
    axs[-1].set_xlabel('Cosine Similarity')

    st.pyplot(fig)


if 'df' not in st.session_state: st.session_state['df'] = None
if 'filtered_data' not in st.session_state: st.session_state['filtered_data'] = None
if 'Применить фильтры' not in st.session_state: st.session_state['Применить фильтры'] = None
if 'semantic_queries_unfilled' not in st.session_state: st.session_state['semantic_queries_unfilled'] = ['']
if 'semantic_queries' not in st.session_state: st.session_state['semantic_queries'] = ['']
if 'apply_filters' not in st.session_state: st.session_state['apply_filters'] = None

    
# < DATA LOADING >
st.markdown('### Выберите файлы для загрузки')
files = glob.glob('saved_data/**/*')

with st.form("data_form"):
    selected_files = st.multiselect('Select files to load', files)
    upload_button = st.form_submit_button('Load data')
    if upload_button:
        if selected_files:
            df = load_to_df(selected_files)
            st.session_state['df'] = df
if st.session_state['df'] is not None:
    df = st.session_state['df']
    st.write('Loaded data:', df.shape, df.sample(1))
# < / DATA LOADING >


if st.session_state['df'] is not None:
    filtered_data = st.session_state['df'].copy()
    show_data_toggle(filtered_data, ROWS_LIMIT)

    # FILTERS
    with st.form("filters_form"):

        st.header("2. Filter data")
        
        st.markdown("Removes duplicated rows, and creates new column 'count'")
        st.session_state.remove_duplicates_checkbox = st.checkbox(label='Remove duplicates', value=True)

        # FTS        
        st.subheader("Strict match filtering")
        st.markdown("Use this separator between words ; (for example - sky;blue)")
        white_list_and = st.text_input("White list (AND) ")
        white_list_or = st.text_input("White list (OR)")
        black_list = st.text_input("Black list")
    
        st.subheader("Semantic filtering")
        for i, condition in enumerate(st.session_state['semantic_queries_unfilled']):
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input(f"Filter on semantic meaning {i+1}", value=condition, key=f"query_{i}")
                semantic_threshold = st.slider("Similarity threshold", 0.0, 1.0, (0.5, 1.0), key=f"semantic_threshold_{i}")
                st.session_state.semantic_queries[i] = (query, semantic_threshold)
            with col2:
                if st.form_submit_button(f'Удалить {i+1}'):
                    remove_condition(i)
            
        if st.form_submit_button('add semantic condition'):
            add_condition()

        apply_filters = st.form_submit_button('Apply filters')
        if apply_filters:
            st.session_state.apply_filters = True
            
    if st.session_state.apply_filters:
        if st.session_state.remove_duplicates_checkbox:
            filtered_data = remove_duplicates(filtered_data)

        if len(st.session_state['semantic_queries']) > 0:
            semantic_queries = [(query, semantic_threshold) for query, semantic_threshold in st.session_state.semantic_queries if len(query) > 0]
            
            # < SEMANTIC SEARCH >
            cos_sim_dict = {}
            for query, (low_threshold, high_threshold) in semantic_queries:
                res = get_embedding_runpod(query)
                target_embedding = res['embeddings']
                cos_sim = get_cos_sim(filtered_data, target_embedding, low_threshold, high_threshold)
                cos_sim_dict[query] = cos_sim
                filtered_data[f'cos_sim {query}'] = cos_sim_dict[query]
                
            idx = 0 
            for query, (low_threshold, high_threshold) in semantic_queries:
                filtered_data = filtered_data[(filtered_data[f'cos_sim {query}'] >= low_threshold) & (filtered_data[f'cos_sim {query}'] <= high_threshold)]
            
            if len(semantic_queries) > 0:
                filtered_data = filtered_data.sort_values(by=f'cos_sim {query}', ascending=False)

            st.write(f"Semantic filtering shape:", filtered_data.shape)
            show_histogram_toggle()
            # < / SEMANTIC SEARCH >

        if len(white_list_and) or len(white_list_or) or len(black_list):
            # < FTS >
            filtered_data = apply_fts(filtered_data, white_list_and, white_list_or, black_list)
            st.write(f"Full text search shape:", filtered_data.shape)
            # < / FTS >

        st.session_state['filtered_data'] = filtered_data
        
        # < SHOW DF >
        show_df(filtered_data, default_columns=['question'], default_max_rows=10000)
        # < / SHOW DF >
        
    # SAVE
    st.sidebar.header("3. Saving")
    save_path = st.sidebar.text_input("Путь и название файла для сохранения", value="saved_data/exp1/filename.feather")

    if st.sidebar.button("Save"):
        if st.session_state['filtered_data'] is None:
            st.sidebar.error(f"Error - no filters applied.")

        if st.session_state['filtered_data'] is not None:
            filtered_data = st.session_state['filtered_data']
            if filtered_data.shape[0] == 0:
                st.sidebar.error(f"Error - filtered data contains 0 rows")

            else:
                directory = "/".join(save_path.split('/')[:-1])
                filename = save_path.split('/')[-1]
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filepath = os.path.join(directory, filename)
                if filename.endswith('.csv'):
                    filtered_data.to_csv(filepath, index=False)
                if filename.endswith('.feather'):
                    filtered_data.to_feather(filepath)

                st.sidebar.success(f"Data saved to {filepath}")


    # Sidebar for LLM parameters
    with st.form("generation_params_form"):
        st.sidebar.markdown('### LLM params')
        model_name = st.sidebar.selectbox("Select a model:", [model["model"] for model in AVAILABLE_MODELS])
        model_creds = next((model for model in AVAILABLE_MODELS if model["model"] == model_name), None)
        max_tokens = st.sidebar.number_input("max_output_tokens", min_value=1, max_value=4096, value=512)
        temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=1.0, value=0.1)

    # LOAD AND SAVE TEMPLATE
    selected_template, prompt_text = templates_form()

    # Button to trigger the LLM call
    num_messages_per_call = st.number_input("question per call", min_value=1, value=50)
    num_llm_calls = st.number_input("Number of LLM calls", min_value=1, value=1)
    max_string_length = st.number_input("Maximum string length (32k symbols)", min_value=1, value=32000)

    if st.button("Call LLM"):
        if prompt_text:
            prompt_template = dedent("""
            You are provided with QUERY from human, and data which you will use to answer it as best as you can according to given QUERY.
            I'm going to tip $1,000,000 for the best reply. 
            Quality of your answer is critical for my career.
            Respond in a natural, human-like manner.

            <QUERY>
            $QUERY
            </QUERY>

            <DATA>
            $DATA
            </DATA>
            """).strip()
            
            data_list = filtered_data['question'].values.tolist()
            data = " \n".join([x for x in data_list])
            total_data_length = len(data)
            
            async def process_llm_calls():
                tasks = []
                for llm_call in range(num_llm_calls):
                    start_idx = llm_call * num_messages_per_call
                    end_idx = start_idx + num_messages_per_call

                    if start_idx >= total_data_length:
                        st.text(f"No more data to process for LLM call {llm_call + 1}")
                        break

                    data_chunk = data_list[start_idx:end_idx]
                    data_chunk_str = " \n".join([x for x in data_chunk])
                    chunk_length = len(data_chunk_str)
                    if chunk_length > max_string_length:
                        data_chunk_str = data_chunk_str[:max_string_length]

                    prompted = prompt_template.replace('$QUERY', str(prompt_text)).replace('$DATA', data_chunk_str)

                    messages = [{'role': 'user', 'content': prompted}]
                    tasks.append(async_call_llm(messages, model_creds, max_tokens, temperature, rate_limiter=RateLimiter(50, 5)))

                results = await asyncio.gather(*tasks)
                for llm_call, result in enumerate(results):
                    result_text = result.get('text_response', 'No response text found')
                    start_idx = llm_call * num_messages_per_call
                    end_idx = start_idx + num_messages_per_call
                    st.write(f"Result {llm_call + 1} (rows {start_idx} to {end_idx}):\nChunk total length: {chunk_length}\n\n", result_text)

            asyncio.run(process_llm_calls())

