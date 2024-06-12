import re, os
import streamlit as st
import pandas as pd
import glob
import json 
import matplotlib.pyplot as plt
import numpy as np

from data_utils import representative_tfidf_ngrams, cosine_similarity_1d, load_to_df, get_cos_sim
from llm_utils import get_embedding_local, get_embedding_runpod

with open('src/config.json', 'r') as f:
    config = json.load(f)

# File Upload Section
st.set_page_config(page_title="Загрузка данных", page_icon="🏠")
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
                df = df[df['questions'].str.contains(term, case=False, na=False)]

    if white_list_or:
        white_terms_or = white_list_or.split(';')
        white_terms_or = [x for x in white_terms_or if len(x) > 0]
        df = df[df['questions'].str.contains('|'.join(white_terms_or), case=False, na=False)]
    
    if black_list:
        black_terms = black_list.split(';')
        for term in black_terms:
            if len(term) > 0:
                df = df[~df['questions'].str.contains(term, case=False, na=False)]
    return df

@st.experimental_fragment
def show_data_toggle(data, rows_limit):
    show_original = st.toggle(label='Показать загруженные данные', key='toggle_data')
    if show_original:
        st.dataframe(data[:rows_limit])

@st.experimental_fragment
def show_histogram_toggle():
    show = st.toggle(label='Показать гистограмму', key='toggle_hist')
    if show:
        plot_histograms(cos_sim_dict)
        
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
    upload_button = st.form_submit_button('Загрузить данные')
    if upload_button:
        if selected_files:
            df = load_to_df(selected_files)
            st.session_state['df'] = df
if st.session_state['df'] is not None:
    df = st.session_state['df']
    st.write('Загруженные данные:', df.shape, df.sample(1))
# < / DATA LOADING >


if st.session_state['df'] is not None:
    filtered_data = st.session_state['df'].copy()
    show_data_toggle(filtered_data, ROWS_LIMIT)

    # FILTERS
    with st.form("filters_form"):

        st.header("2. Фильтрация")
        st.subheader("Полно-текстовая фильтрация")
        st.markdown("* Используйте разделитель между словами ; (например sky;blue)")
    
        white_list_and = st.text_input("Белый список (AND) ")
        white_list_or = st.text_input("Белый список (OR)")
        black_list = st.text_input("Черный список")
    
        st.subheader("Семантическая фильтрация")
        for i, condition in enumerate(st.session_state['semantic_queries_unfilled']):
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input(f"Отфильтровать по смыслу {i+1}", value=condition, key=f"query_{i}")
                semantic_threshold = st.slider("Порог похожести", 0.0, 1.0, (0.5, 1.0), key=f"semantic_threshold_{i}")
                st.session_state.semantic_queries[i] = (query, semantic_threshold)
            with col2:
                if st.form_submit_button(f'Удалить {i+1}'):
                    remove_condition(i)
            
        if st.form_submit_button('Добавить условие'):
            add_condition()

        apply_filters = st.form_submit_button('Применить фильтры')
        if apply_filters:
            st.session_state.apply_filters = True
            
    if st.session_state.apply_filters:
        if len(st.session_state['semantic_queries']) > 0:
            semantic_queries = [(query, semantic_threshold) for query, semantic_threshold in st.session_state.semantic_queries if len(query) > 0]
            
            print('semantic_queries', semantic_queries)
            # < SEMANTIC SEARCH >
            cos_sim_dict = {}
            for query, (low_threshold, high_threshold) in semantic_queries:
                if config.get('use_local_emb_model'):
                    res = get_embedding_local(query)
                else:
                    res = get_embedding_runpod(query)
                target_embedding = res['embeddings']
                cos_sim = get_cos_sim(filtered_data, target_embedding, low_threshold, high_threshold)
                cos_sim_dict[query] = cos_sim
                filtered_data[f'cos_sim {query}'] = cos_sim_dict[query]
                
            idx = 0 
            for query, (low_threshold, high_threshold) in semantic_queries:
                filtered_data = filtered_data[(filtered_data[f'cos_sim {query}'] >= low_threshold) & (filtered_data[f'cos_sim {query}'] <= high_threshold)]
            filtered_data = filtered_data.sort_values(by=f'cos_sim {query}', ascending=False)

            st.write(f"Semantic filtering shape:", filtered_data.shape)
            show_histogram_toggle()
            # < / SEMANTIC SEARCH >

        if len(white_list_and) or len(white_list_or) or len(black_list):
            # < FTS >
            filtered_data = apply_fts(filtered_data, white_list_and, white_list_or, black_list)
            st.write(f"Full text search shape:", filtered_data.shape)
            # < / FTS >

        st.dataframe(filtered_data[:ROWS_LIMIT])
        st.session_state['filtered_data'] = filtered_data


    # SAVE
    st.sidebar.header("3. Сохранение")
    save_path = st.sidebar.text_input("Путь и название файла для сохранения", value="saved_data/exp1/filename.feather")

    if st.sidebar.button("Сохранить"):
        if st.session_state['filtered_data'] is None:
            st.sidebar.error(f"Ошибка. Прежде чем сохранять - примените фильтры.")

        if st.session_state['filtered_data'] is not None:
            filtered_data = st.session_state['filtered_data']
            if filtered_data.shape[0] == 0:
                st.sidebar.error(f"Ошибка. Отфильтрованный датасет содержит 0 строчек")

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

                st.sidebar.success(f"Даннвые с выбранными фильтрами успешно сохранены в {filepath}")


