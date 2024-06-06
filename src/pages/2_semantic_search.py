import re, os
import streamlit as st
import pandas as pd
import glob
import json 

from data_utils import load_h5_to_dataframe, representative_tfidf_ngrams, cosine_similarity_1d, load_to_df, semantic_search
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

@st.cache_data
def load_data(uploaded_file):
    data_load_state = st.text('Loading data...')

    data = pd.read_csv(uploaded_file)
    if 'question' not in data.columns.tolist():
        st.error('data should contain column: sender')
    elif 'answer' not in data.columns.tolist():
        st.error('data should contain column: body')
    data_load_state.text('Loading data...done!')
    data = data.dropna(subset=['sender', 'body'])
    return data

@st.cache_data
def apply_fts(filtered_data, white_list, black_list):

    if len(white_list):
        white_list_words = white_list.split(';')
        fts_pattern = '|'.join(map(re.escape, white_list_words))
        filtered_data = filtered_data[filtered_data['questions'].str.contains(fts_pattern, case=False, regex=True)]
    print('1', filtered_data.shape)

    if len(black_list):
        black_list_words = black_list.split(';')
        blk_pattern = '|'.join(map(re.escape, black_list_words))
        filtered_data = filtered_data[~filtered_data['questions'].str.contains(blk_pattern, case=False, regex=True)]

    print('3', filtered_data.shape)
    return filtered_data

@st.experimental_fragment
def show_data_toggle(data, rows_limit):
    show_original = st.toggle(label='Показать загруженные данные', key='toggle_data')
    if show_original:
        st.dataframe(data[:rows_limit])


if 'df' not in st.session_state: st.session_state['df'] = None
if 'filtered_data' not in st.session_state: st.session_state['filtered_data'] = None
if 'Применить фильтры' not in st.session_state: st.session_state['Применить фильтры'] = None

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
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    st.write('Загруженные данные:', df.shape, df.sample(1))
# < / DATA LOADING >

# LOGGING

# show_original = st.checkbox('Показать загруженные данные')
# if show_original:
#     st.dataframe(data[:ROWS_LIMIT])

if 'df' in st.session_state and st.session_state['df'] is not None:
    filtered_data = st.session_state['df'].copy()
    show_data_toggle(filtered_data, ROWS_LIMIT)

    # FILTERS
    with st.form("filters_form"):

        st.header("2. Фильтрация")
        st.subheader("Полно-текстовая фильтрация")
        white_list = st.text_input("Белый список (используйте разделитель ;)")
        black_list = st.text_input("Черный список (используйте разделитель ;)")

        st.subheader("Семантическая фильтрация")
        semantic_query = st.text_input("Отфильтровать по смыслу")
        semantic_threshold = st.slider("Порог похожести", 0.0, 1.0, 0.7)
        apply_filters = st.form_submit_button('Применить фильтры')

    if apply_filters:
        if isinstance(semantic_query, str) and len(semantic_query) > 1:
            print('semantic_query', semantic_query)
            # < SEMANTIC SEARCH >
            if config.get('use_local_emb_model'):
                res = get_embedding_local(semantic_query)
            else:
                res = get_embedding_runpod(semantic_query)
            target_embedding = res['embeddings']

            filtered_data = semantic_search(filtered_data, target_embedding)
            filtered_data = filtered_data[filtered_data['cos_sim'] > semantic_threshold]
            st.write(f"Semantic filtering shape:", filtered_data.shape)
            # < / SEMANTIC SEARCH >

        if len(white_list) or len(black_list):
            # < FTS >
            filtered_data = apply_fts(filtered_data, white_list, black_list)
            st.write(f"Full text search shape:", filtered_data.shape)
            # < / FTS >

        st.dataframe(filtered_data[:ROWS_LIMIT])
        st.session_state['filtered_data'] = filtered_data


    # SAVE
    st.sidebar.header("3. Сохранение")
    save_path = st.sidebar.text_input("Путь и название файла для сохранения", value="saved_data/test/sample.csv")

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
                filtered_data.to_csv(filepath, index=False)
                st.sidebar.success(f"Даннвые с выбранными фильтрами успешно сохранены в {filepath}")


