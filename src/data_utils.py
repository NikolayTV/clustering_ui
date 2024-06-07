from collections import Counter
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from llm_utils import get_embedding_local, get_embedding_runpod


# Функция для извлечения n-грамм, взвешенных по TF-IDF
def extract_tfidf_ngrams(series, n=1):
    vectorizer = TfidfVectorizer(ngram_range=(n, n))
    tfidf_matrix = vectorizer.fit_transform(series)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    return Counter(dict(zip(feature_names, tfidf_scores)))

def representative_tfidf_ngrams(df, n=1, total_ngrams=20):
    """
    Извлечение репрезентативных n-грамм для каждого кластера с учетом TF-IDF
    """

    clusters = df['cluster'].unique()
    representative_ngrams = {}
    for cluster in clusters:
        cluster_texts = df[df['cluster'] == cluster]['questions']
        other_texts = df[df['cluster'] != cluster]['questions']
        
        # Объединение текстов для правильного расчета TF-IDF
        combined_texts = pd.concat([cluster_texts, other_texts])
        
        # Извлечение n-грамм, взвешенных по TF-IDF
        ngram_counter = extract_tfidf_ngrams(combined_texts, n=n)
        
        # Извлечение наиболее важных n-грамм для текущего кластера
        cluster_ngrams = extract_tfidf_ngrams(cluster_texts, n=n)
        most_common_ngrams = {k: cluster_ngrams[k] * ngram_counter[k] for k in cluster_ngrams}
        representative_ngrams[cluster] = Counter(most_common_ngrams).most_common(total_ngrams)

    repr_words = {}
    for cluster_id, value in representative_ngrams.items():
        if cluster_id not in repr_words:
            repr_words[cluster_id] = ''

        for num, i in enumerate(value):
            repr_words[cluster_id] += i[0] + ", "
            if num > total_ngrams:
                break

    return repr_words


@st.cache_data
def load_to_df(selected_files):
    dataframes = []
    progress_bar = st.progress(0)
    
    for i, file_path in enumerate(selected_files):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            if file_path.endswith('.feather'):
                df = pd.read_feather(file_path)

            dataframes.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(selected_files))
    
    if dataframes:
        big_dataframe = pd.concat(dataframes, ignore_index=True)
        big_dataframe = big_dataframe[['questions', 'answers', 'dates', 'embeddings']]
        return big_dataframe
    else:
        st.write('No valid data to display.')
        return None

def cosine_similarity_1d(array1, array2):
    dot_product = np.sum(array1 * array2, axis=1)
    norm1 = np.linalg.norm(array1, axis=1)
    norm2 = np.linalg.norm(array2, axis=1)

    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

@st.cache_data
def semantic_search(df, target_embedding):
    """
    Sorts df based on cos sim
    """
    start_time = time.time()

    embeddings = np.array(df['embeddings'].tolist())
    cos_sim = cosine_similarity(np.array(target_embedding).reshape(1, -1), embeddings).flatten()
    df['cos_sim'] = cos_sim
    df = df.sort_values(by='cos_sim', ascending=False)
    execution_time = time.time() - start_time
    st.write(f'Semantic search execution time: {round(execution_time,2)} seconds')
    return df
