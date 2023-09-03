import streamlit as st
import pandas as pd

from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull, save_model, load_model
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models

# import plotly.express as px
# from pycaret.regression import setup, compare_models, pull, save_model, load_model
# from pycaret.classification import *
import os

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = pd.DataFrame() # default dataframe if one has not been provided

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Autopycaret")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    task = st.radio('Select Task', ['Regression', 'Classification'])

    if st.button('Run Modelling'):
        if task == 'Regression':
            setup_df = regression_setup(df, target=chosen_target, silent=True)
        elif task == 'Classification':
            setup_df = classification_setup(df, target=chosen_target, silent=True)
        
        st.dataframe(setup_df)

        if task == 'Regression':
            best_model = regression_compare_models()
        elif task == 'Classification':
            best_model = classification_compare_models()
        
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model has been saved yet. Please run modelling first.")