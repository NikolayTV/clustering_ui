import os

import streamlit as st

from data_utils import load_templates, save_template

@st.experimental_fragment
def show_df(df, default_columns=['question'], default_max_rows=10000):
    SELECTED_COLUMNS = st.multiselect('Columns to display', df.columns.tolist(), default=default_columns)
    ROWS_LIMIT = st.number_input("Rows limit", min_value=0, value=default_max_rows)
    styled_df = df[:ROWS_LIMIT][SELECTED_COLUMNS].reset_index(drop=True)
    st.dataframe(styled_df, width=1000)
    
@st.cache_data
def remove_duplicates(df):
    df['count'] = df.groupby('question')['question'].transform('count')
    df = df.drop_duplicates(subset='question')
    return df

@st.experimental_fragment
def templates_form(default_template=None):
    if 'show_save_form' not in st.session_state:
        st.session_state.show_save_form = False

    templates = load_templates(templates_dir='saved_templates')

    selected_template = st.selectbox("Select a prompt template", list(templates.keys()), index=list(templates.keys()).index(default_template))
    prompt_text = st.text_area("Prompt", templates[selected_template].strip(), height=300)
    if st.button("Save template"):
        st.session_state.show_save_form = True  
        st.session_state.prompt_text = prompt_text 
        st.session_state.selected_template = selected_template
        # st.experimental_rerun()
            
    if st.session_state.show_save_form:
        with st.form("save_form"):
            new_template_name = st.text_input("Enter new path (it should end with .txt)", value=st.session_state.selected_template)
            if st.button("Confirm path and save"):
                save_template(new_template_name, st.session_state.prompt_text)
                st.session_state.show_save_form = False  
    
    return selected_template, prompt_text