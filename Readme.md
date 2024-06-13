 
# This is UI for interactive chat analysis and clustering with streamlit
## Installation
* Install nlp package for lemmatization:
$ python -m spacy download en_core_web_sm

## Run embedding inference server
$ python -u large-gte-eng/src/handler.py --rp_serve_api 

## Run streamlit app
$ streamlit run src/1_home.py

## For this thing to work you need to prepare text data with embeddings and place it in the root of repo on ./saved_data folder
script to prepare data - notebooks/prepare_dataset.ipynb


## TODO

% * добавить эмоции

## Done
* advanced FTS and layered semantic filtering
* todo добавить фильтры языка
* интеграция с feather
* рабочий пайплайн семантического поиска
* рабочий пайплайн кластеризации
* добавить лемматизацию слов для tfidf
* Добавить конфиг с выбором моделей (словарь с названием и url)
* дропаем дубликаты (возможно до кластеризации, но в отображении показывать частотность данной фразы) (добавил уборку дубликатов и добавление колонки count)
* выбор колонок и строк в датафрейме
* Save and load prompt templates
* Limit number of symbols to prevent max input tokens error (naive approach with symbols)
* Specify how many and how big chunks for llm call in semantic search
* добавить сколько уникальных юзеров и чатов в кластере
