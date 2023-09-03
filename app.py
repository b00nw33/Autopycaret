import streamlit as st
import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
from pycaret.classification import *
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
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
