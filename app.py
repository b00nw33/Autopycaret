import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull, save_model, load_model
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models

def load_dataset(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

def upload_dataset():
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)
        return df
    return None

def perform_eda(df):
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

def run_modelling(df, chosen_target, task):
    st.dataframe(setup_df)
    best_model = None
    if task == 'Regression':
        setup_df = regression_setup(df, target=chosen_target, silent=True)
        best_model = regression_compare_models()
    elif task == 'Classification':
        setup_df = classification_setup(df, target=chosen_target, silent=True)
        best_model = classification_compare_models()

    compare_df = pull()
    st.dataframe(compare_df)
    save_model(best_model, 'best_model')

def download_model():
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model has been saved yet. Please run modelling first.")

def main():
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Autopycaret")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

    df = load_dataset('dataset.csv')

    if choice == "Upload":
        df = upload_dataset()
    elif choice == "Profiling" and not df.empty:
        perform_eda(df)
    elif choice == "Modelling" and not df.empty:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        task = st.radio('Select Task', ['Regression', 'Classification'])
        if st.button('Run Modelling'):
            run_modelling(df, chosen_target, task)
    elif choice == "Download":
        download_model()

if __name__ == "__main__":
    main()
