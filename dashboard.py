import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_recall_curve

df = pd.read_csv("creditcard.csv")

# Setting hyperparameters, feature selection

st.title("Credit Card Fraud Detection Model Benchmarking")
scale = st.checkbox("Choose to scale the data: ")
if (scale):
    st.write("Scaling data with standard scaler")
else:
    st.write("Not scaling data")
model_select = st.radio("Choose options: ", ("Logit", "Random Forest", "Decision Tree", "Support Vector Machine"))
st.write("Selected", model_select, " as model")
select = st.selectbox("Choose evaluation metric: ", ['Precision', 'Recall', 'Precision-Recall'])
st.write("Selected ", select, " as metric")
many_select = st.multiselect("Options: ", df.columns[:-1].tolist())
st.write("Selected ", len(many_select), " features")
model_name = st.text_input("Enter model name / id",  "Name ...")
st.write("Model to be named ", model_name)
slide = st.slider("Select level", 1, 10)
st.write("Currently at level: ", slide)

