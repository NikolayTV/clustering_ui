import re, os
import streamlit as st
import pandas as pd

# File Upload Section
st.set_page_config(page_title="Загрузка данных", page_icon="🏠")
st.title("Интерактивная аналитика & тематическое моделирование чатов")

ROWS_LIMIT = st.sidebar.text_input("Максимальное количество строк для отображения", value=100)
try:
    ROWS_LIMIT = int(ROWS_LIMIT)
except ValueError:
    st.error("Пожалуйста, введите числовое значение для максимального количества строк")
    ROWS_LIMIT = 100



@st.cache_data
def load_data(uploaded_file):
    data_load_state = st.text('Loading data...')

    data = pd.read_csv(uploaded_file)
    if 'sender' not in data.columns.tolist():
        st.error('data should contain column: sender')
    elif 'body' not in data.columns.tolist():
        st.error('data should contain column: body')

    data_load_state.text('Loading data...done!')

    # preprocess
    data = data.dropna(subset=['sender', 'body'])

    return data

@st.cache_data
def apply_fts(filtered_data, white_list, black_list):

    if len(white_list):
        white_list_words = white_list.split(';')

        fts_pattern = '|'.join(map(re.escape, white_list_words))
        filtered_data = filtered_data[filtered_data['body'].str.contains(fts_pattern, case=False, regex=True)]
    print('1', filtered_data.shape)

    if len(black_list):
        black_list_words = black_list.split(';')
        blk_pattern = '|'.join(map(re.escape, black_list_words))
        filtered_data = filtered_data[~filtered_data['body'].str.contains(blk_pattern, case=False, regex=True)]

    print('3', filtered_data.shape)
    return filtered_data


st.sidebar.header("1. Загрузка файла")
uploaded_file = st.sidebar.file_uploader("Выберите ваш файл в формате .csv", type="csv")
if uploaded_file:
    if 'data' not in st.session_state: st.session_state['data'] = None
    if 'filtered_data' not in st.session_state: st.session_state['filtered_data'] = None
    if 'Применить фильтры' not in st.session_state: st.session_state['Применить фильтры'] = None

    if st.session_state['data'] is None:
        st.session_state['data'] = load_data(uploaded_file)
    
    data = st.session_state['data']
    print('asdas', data.shape)
    # LOGGING
    st.write(f"В исходных данных содержится {data.shape[0]} строчек и {data.shape[1]} столбцов")

    show_original = st.checkbox('Показать загруженные данные')
    if show_original:
        st.dataframe(data[:ROWS_LIMIT])

    # FILTRES
    st.sidebar.header("2. Фильтрация")
    st.sidebar.subheader("Полно-текстовая фильтрация")
    white_list = st.sidebar.text_input("Белый список (используйте разделитель - ;)")
    black_list = st.sidebar.text_input("Черный список (используйте разделитель - ;)")

    # Semantic search
    semantic_query = st.sidebar.text_input("Отфильтровать по смыслу")
    semantic_threshold = st.sidebar.slider("Порог похожести", 0.0, 1.0, 0.7)


    # Apply Filters Button
    if st.sidebar.button("Применить фильтры") or st.session_state['Применить фильтры']:
        st.session_state['Применить фильтры'] = True
        filtered_data = st.session_state['data'].copy()
        print('filtered_data adsa', filtered_data.shape)
        
        filtered_data = apply_fts(filtered_data, white_list, black_list)
        st.write(f"Размерность данных после full-text-search:", filtered_data.shape)

        if semantic_query:
            st.write(f"Applying semantic search with query: {semantic_query} and semantic_threshold: {semantic_threshold}")
            filtered_data = filtered_data
        

        st.write(f"В отфильтрованных данных содержится {filtered_data.shape[0]} строк")
        st.dataframe(filtered_data[:ROWS_LIMIT])
        st.session_state['filtered_data'] = filtered_data


    # SAVE
    save_path = st.sidebar.text_input("Путь и название файла для сохранения", value="saved_data/test/sample.csv")
    if st.sidebar.button("Сохранить") and st.session_state['data'] is not None:
        filtered_data = st.session_state['filtered_data']
        if filtered_data.shape[0] == 0:
            st.write(f"Сначала примените фильтры. Отфильтрованный датасет содержит {filtered_data.shape[0]} строчек")

        else:
            directory = "/".join(save_path.split('/')[:-1])
            filename = save_path.split('/')[-1]
            if not os.path.exists(directory):
                os.makedirs(directory)

            filepath = os.path.join(directory, filename)
            filtered_data.to_csv(filepath, index=False)
            st.sidebar.success(f"Даннвые с выбранными фильтрами успешно сохранены в {filepath}")


