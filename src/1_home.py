import streamlit as st
from llm_utils import load_model


st.set_page_config(page_title="Home", page_icon="🏠")
st.title("Тематическое моделирование чатов")

st.write("""1 - Загрузите документы""")
st.write("""2 - Используйте фильтры что бы создать сегменты, проанализировать и сохранить для дальнейшего использования""")
st.write("""3 - Кластеризуйте ваш документ целиком или по выбранным сегментам""")


st.write("##### You can find the full codebase & project specifics on my [GitHub](https://github.com/).")



