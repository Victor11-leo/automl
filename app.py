import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup,compare_models,pull,save_model
from pycaret.regression import setup as setupR,compare_models as compare_modelsR,pull as pullR,save_model as save_modelR


with st.sidebar:
    st.image("https://plus.unsplash.com/premium_photo-1683121710572-7723bd2e235d?w=1000&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8bWFjaGluZSUyMGxlYXJuaW5nfGVufDB8fDB8fHww")
    st.title("AutoML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("Build an ML pipeline in seconds. Magical stuff")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choice == "Upload":
    st.title("Upload your Data for Modelling")
    st.info("Accepts csv files only")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated EDA")
    profile_report = ProfileReport(df,title="Profiling Report")
    st_profile_report(profile_report)

if choice == "ML":
    st.title("ML quick")
    target = st.selectbox("Select your target",df.columns)
    type = st.selectbox("Select your ML type",['Classification','Regression'])

        

    if st.button("Run modelling"):

        if (type == 'Classification'):
            with st.spinner("Setting up ML experiment"):
                setup(df,target=target)
            setup_df = pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)

            with st.spinner("Comparing models please wait"):
                best_model = compare_models()

            compare_df = pull()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            save_model(best_model,'best_model')

        if (type == 'Regression'):
            with st.spinner("Setting up ML experiment"):
                setupR(df,target=target)
            setup_df = pullR()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)

            with st.spinner("Comparing models please wait"):
                best_model = compare_modelsR()

            compare_df = pullR()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            save_modelR(best_model,'best_model')

if choice == "Download":
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")