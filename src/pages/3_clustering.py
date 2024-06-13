from textwrap import dedent 
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
from llm_utils import async_name_with_llm, RateLimiter
from prompts import clustering_system_message
from config import AVAILABLE_MODELS
from streamlit_utils import show_df
from streamlit_utils import templates_form

# File Upload Section
st.set_page_config(
    page_title="Clustering",
    page_icon="🌲🌼🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Chat topic modeling")

rate_limiter = RateLimiter(50, 5)

NUMBER_OF_DOCS_TO_USE_IN_NAMING = 20
KMEANS_NITER = 100
MAX_SYMBOLS_PER_MESSAGE = 400
DENDRO_COLOR_THRESHOLD = 0.8

@st.cache_data
def do_clustering(df, n_clusters, tfidf_checkbox):
    embeddings = np.array(df['embeddings'].values.tolist())
    embeddings = np.array(df['embeddings'].values.tolist())
    kmeans = faiss.Kmeans(embeddings.shape[1], n_clusters, niter=KMEANS_NITER, verbose=True, nredo=1, seed=42)
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

    st.text('Step 2 - extracting keywords')
    trigrams = {}
    if tfidf_checkbox:
        try:
            trigrams = representative_tfidf_ngrams(df, n=2)
        except:
            st.text('Cannot extract keywords - not enough data. Skipping')
            trigrams = ''
            
    return df, topics_df, trigrams


# In parallel processes all tokens
async def name_topic(topic_n, sampled_documents_df, trigrams, topics_df, system_message, model_creds, max_tokens, temperature):
    sampled_documents_df = sampled_documents_df.groupby('cluster').head(NUMBER_OF_DOCS_TO_USE_IN_NAMING)
    documents = [x[:MAX_SYMBOLS_PER_MESSAGE]+"\n" for x in sampled_documents_df[sampled_documents_df['cluster'] == topic_n]['question'].values.tolist()]
    keywords = trigrams.get(topic_n)
    cluster_name = await async_name_with_llm(documents, keywords, system_message=system_message, model_creds=model_creds, max_tokens=max_tokens, temperature=temperature, rate_limiter=rate_limiter)

    idx = topics_df[topics_df['cluster'] == topic_n].index[0]
    topics_df.loc[idx, 'cluster_name'] = cluster_name['text_response']
    print(f'Topic: {topic_n} Name:', cluster_name['text_response'])

async def name_all_topics(sampled_documents_df, trigrams, topics_df, system_message, model_creds, max_tokens, temperature):
    tasks = []
    for topic_n in sampled_documents_df['cluster'].unique()[:]:
        tasks.append(name_topic(topic_n, sampled_documents_df, trigrams, topics_df, system_message, model_creds, max_tokens, temperature))
    await asyncio.gather(*tasks)
    
    
def clustering_pipeline(df, n_clusters, system_message, n_docs_to_show, tfidf_checkbox, model_creds, max_tokens, temperature):

    st.text('Step 1 - START training')
    # CLUSTERING
    
    df, topics_df, trigrams = do_clustering(df, n_clusters, tfidf_checkbox)
    
    def sample_from_top_n(group, n_top=n_docs_to_show):
        return group.head(n_top)

    top_documents_df = df.sort_values(by=['cluster', 'distance_to_centroids'], ascending=[True, False])
    top_documents_df = top_documents_df.drop_duplicates(subset=['question'])
    sampled_documents_df = top_documents_df.groupby('cluster').apply(sample_from_top_n).reset_index(drop=True)
    sampled_documents_df = sampled_documents_df[['cluster', 'question', 'answer', 'distance_to_centroids']]
    
    # <LLM NAMING>
    st.text('Step 3 - naming clusters')
    asyncio.run(name_all_topics(sampled_documents_df, trigrams, topics_df, system_message, model_creds, max_tokens, temperature))
    
    if 'userId' not in df.columns: df['userId'] = None
    if 'chatId' not in df.columns: df['chatId'] = None
    # Group by 'cluster' and calculate unique counts
    grouped = df.groupby('cluster').agg(
        unique_userIds=('userId', 'nunique'),
        unique_chatIds=('chatId', 'nunique')
    ).reset_index()

    # Merge unique counts into topics_df
    topics_df = topics_df.merge(grouped, on='cluster', how='left')

    # Calculate cluster counts and percentages
    cluster_counts = df['cluster'].value_counts().to_dict()
    topics_df['cluster_cnt'] = topics_df['cluster'].map(cluster_counts)
    topics_df['cluster_pct'] = (topics_df['cluster_cnt'] / df.shape[0]).round(4)
    
    topics_df['cluster_title'] = topics_df.apply(
        lambda x: f"{x['cluster']}: {x['cluster_name']}: {x['cluster_pct']*100:.2f}%, {x['cluster_cnt']}#, {x['unique_userIds']} users, {x['unique_chatIds']} chats",
        axis=1
    )

    # </LLM NAMING>

    distance_function = lambda x: 1 - cosine_similarity(x)
    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    fig = ff.create_dendrogram(np.array(topics_df['centroids'].values.tolist()),
                               hovertext=topics_df['cluster_title'].values.tolist(),
                               orientation='left',
                               linkagefun=linkage_function,
                               distfun=distance_function,
                               color_threshold=DENDRO_COLOR_THRESHOLD)
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
    st.text('Step 4 - creating plot')
    st.plotly_chart(fig)
    return fig, topics_df, sampled_documents_df, df



if 'df' not in st.session_state: st.session_state['df'] = None
if "prompt_template" not in st.session_state: st.session_state.prompt_template = clustering_system_message

# < DATA LOADING >
st.markdown('### Choose file for loading')
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
    st.write('Loaded data:', df.shape, df.sample(1))
# < / DATA LOADING >

    # < PROMPT SELECTION >
    st.markdown('### LLM instructions for clusters naming')
    selected_template, prompt_text = templates_form(default_template='saved_templates/clustering/base.txt')
    clustering_system_message = str(prompt_text).strip()
    # < / PROMPT SELECTION >


st.sidebar.header("Clustering params")
n_clusters = st.sidebar.number_input("Clusters number", min_value=2, value=2)
tfidf_checkbox = st.sidebar.checkbox(label='Use tf-idf keywords for naming', value=False)
n_docs_to_show = st.sidebar.number_input("Example docs to output", value=20)

# Sidebar for LLM parameters
st.sidebar.markdown('### LLM params')
model_name = st.sidebar.selectbox("Select a model:", [model["model"] for model in AVAILABLE_MODELS])
model_creds = next((model for model in AVAILABLE_MODELS if model["model"] == model_name), None)
max_tokens = st.sidebar.number_input("max_output_tokens", min_value=1, max_value=4096, value=50)
temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=1.0, value=0.1)

if st.sidebar.button("Run clustering pipline"):
    fig, topics_df, sampled_documents_df, df = clustering_pipeline(
        df, 
        n_clusters=n_clusters, 
        system_message=clustering_system_message, 
        n_docs_to_show=n_docs_to_show, tfidf_checkbox=tfidf_checkbox,
        model_creds=model_creds, max_tokens=max_tokens, temperature=temperature)
    
    show_df(sampled_documents_df, default_columns=['cluster', 'question'])



