import streamlit as st
import pandas as pd
# import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
# import pandas_profiling
# from pycaret.classification import *
# from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = pd.DataFrame() # default dataframe if one has not been provided

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("OperationalML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
