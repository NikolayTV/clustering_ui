 
# This is UI for interactive chat analysis and clustering with streamlit


## Run embedding inference server
$ python -u large-gte-eng/src/handler.py --rp_serve_api 

## Run streamlit app
$ streamlit run src/1_home.py

## For this thing to work you need to prepare text data with embeddings and place it in the root of repo on ./saved_data folder
script to prepare data - notebooks/prepare_dataset.ipynb


## TODO

* добавить эмоции
* какие колонки отображать чекбоксом
* инструмент работы с таблицами в стремлите
* добавить фильтр даты


## Done
* todo добавить фильтры языка
* интеграция с feather
* рабочий пайплайн семантического поиска
* рабочий пайплайн кластеризации