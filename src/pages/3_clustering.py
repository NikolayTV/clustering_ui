import os, glob

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import streamlit as st
import faiss

import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

from data_utils import representative_tfidf_ngrams, cosine_similarity_1d, load_to_df
from llm_utils import async_name_with_llm, name_with_llm, RateLimiter


# File Upload Section
st.set_page_config(page_title="Кластеризация", page_icon="🏠")
st.title("Тематическое моделирование чатов")

rate_limiter = RateLimiter(50, 5)


def clustering_pipeline(df, n_clusters, batch_size, n_init):

    st.text('Step 1 - START training')

    embeddings = np.array(df['embeddings'].values.tolist())

    ## sklearn
    # k_means = MiniBatchKMeans(
    #     init="k-means++",
    #     n_clusters=n_clusters,
    #     batch_size=batch_size,
    #     n_init=n_init,
    #     max_no_improvement=10,
    #     verbose=0,
    # )
    # k_means.fit(embeddings)
    # st.text('Step 2 - predicting')
    # df['cluster'] = k_means.predict(embeddings)
    # centroids = k_means.cluster_centers_

    ## faiss
    ncentroids = 10
    niter = 50
    verbose = True
    embeddings = np.array(df['embeddings'].values.tolist())
    kmeans = faiss.Kmeans(embeddings.shape[1], n_clusters, niter=n_init, verbose=False, nredo=1, seed=42)
    kmeans.train(embeddings)
    D, I = kmeans.index.search(embeddings, 1)
    df['cluster'] = I.reshape(-1)
    centroids = kmeans.centroids


    topics_df = pd.DataFrame(columns=['cluster', 'cluster_name', 'centroids'])
    topics_df['cluster'] = list(range(len(centroids)))
    topics_df['centroids'] = centroids.tolist()

    centroid_emb_dict = {i: x for i, x in enumerate(centroids)}
    centroid_embeddings = np.array([centroid_emb_dict[i] for i in df['cluster'].values.tolist()])

    distance_to_centroids = cosine_similarity_1d(embeddings, centroid_embeddings)
    df['distance_to_centroids'] = distance_to_centroids

    def sample_from_top_n(group, n_top=20):
        return group.head(n_top)

    top_documents_df = df.sort_values(by=['cluster', 'distance_to_centroids'], ascending=[True, False])
    top_documents_df = top_documents_df.drop_duplicates(subset=['questions'])
    sampled_documents_df = top_documents_df.groupby('cluster').apply(sample_from_top_n).reset_index(drop=True)
    sampled_documents_df = sampled_documents_df[['cluster', 'questions', 'answers', 'distance_to_centroids']]
    st.text('Step 3 - extracting keywords')

    trigrams = representative_tfidf_ngrams(df, n=3)

    # <LLM NAMING>
    st.text('Step 4 - naming clusters')

    # In parallel processes all tokens
    async def process_topic(topic_n, sampled_documents_df, trigrams, topics_df):
        documents = [x[:400] for x in sampled_documents_df[sampled_documents_df['cluster'] == topic_n]['questions'].values.tolist()]
        keywords = trigrams[topic_n]
        cluster_name = await async_name_with_llm(documents, keywords, rate_limiter=rate_limiter)

        idx = topics_df[topics_df['cluster'] == topic_n].index[0]
        topics_df.loc[idx, 'cluster_name'] = cluster_name['text_response']
        print(f'Topic: {topic_n} Name:', cluster_name['text_response'])

    async def process_all_topics(sampled_documents_df, trigrams, topics_df, rate_limiter):
        tasks = []
        for topic_n in sampled_documents_df['cluster'].unique()[:]:
            tasks.append(process_topic(topic_n, sampled_documents_df, trigrams, topics_df))
        await asyncio.gather(*tasks)
        
    asyncio.run(process_all_topics(sampled_documents_df, trigrams, topics_df, rate_limiter))
    cluster_counts = df['cluster'].value_counts().to_dict()
    topics_df['cluster_pct'] = (topics_df['cluster'].replace(cluster_counts)  / df.shape[0]).round(2)
    topics_df['cluster_title'] = topics_df['cluster'].astype(str) + ":" + topics_df['cluster_name'].astype(str) + ":" + topics_df['cluster_pct'].astype(str)
    # </LLM NAMING>

    distance_function = lambda x: 1 - cosine_similarity(x)
    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    st.text('Step 5 - creating plot')

    color_threshold = 0.4
    fig = ff.create_dendrogram(np.array(topics_df['centroids'].values.tolist()),
                               hovertext=topics_df['cluster_title'].values.tolist(),
                               orientation='left',
                               linkagefun=linkage_function,
                               color_threshold=color_threshold)
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': "Cluster hierarchy",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="White")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    width = 1500
    fig.update_layout(height=200 + (15 * topics_df.shape[0]),
                      width=width,
                      yaxis=dict(tickmode="array",
                                 ticktext=topics_df['cluster_title'].values.tolist()))
    y_max = max([trace['y'].max() + 5 for trace in fig['data']])
    y_min = min([trace['y'].min() - 5 for trace in fig['data']])
    st.text('Step 6 - FINISH')
    st.plotly_chart(fig)
    return fig, topics_df, sampled_documents_df, df



if 'df' not in st.session_state: st.session_state['df'] = None

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


# < CLUSTERING >
st.sidebar.header("Параметры обучения")
n_clusters = st.sidebar.number_input("Количество кластеров", min_value=2, value=2)

if st.sidebar.button("Обучить"):
    batch_size = 100
    n_init = 10
    fig, topics_df, sampled_documents_df, df = clustering_pipeline(df, n_clusters=n_clusters, batch_size=batch_size, n_init=n_init)
    st.write(sampled_documents_df[:10000])
# < /CLUSTERING >



