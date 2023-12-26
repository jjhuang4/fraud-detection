import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_recall_curve

df = pd.read_csv("creditcard.csv")

st.title("Credit Card Fraud Detection Model Benchmarking")
scale = st.checkbox("Scale the data: ")
test = st.radio("Choose options: ", ("Train", "Test"))
select = st.selectbox("Options: ", ['A', 'B', 'C'])
many_select = st.multiselect("Options: ", ['A', 'B', 'C'])
txt = st.text_input("Enter some value here",  "Please enter ...")
slide = st.slider("Select level", 1, 10)

