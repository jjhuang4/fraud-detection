import streamlit as st
import pandas as pd
import numpy as np
import os

# Reading in dataset
df = pd.read_csv("creditcard.csv")

# ------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, precision_recall_fscore_support
from sklearn.metrics import roc_curve, RocCurveDisplay
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#features = list(df.columns[:-1])
def train_logistic_model(name, scale, rate, features):
    X = df[features]
    y = df['Class']
    #print(X.shape)
    #print(y.shape)
    undersample = NearMiss(sampling_strategy=rate, version=1, n_neighbors=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_resample, y_train_resample = undersample.fit_resample(X_train, y_train)

    if (scale):
        scaler = StandardScaler().fit(X_train_resample)
        X_train_resample = scaler.transform(X_train_resample)
        X_test = scaler.transform(X_test)

    logit = LogisticRegression(class_weight="balanced").fit(X_train_resample, y_train_resample)

    y_prob_pred = logit.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred)
    opt_idx = np.argmax(tpr-fpr)
    opt_threshold = thresholds[opt_idx]

    y_pred = np.array([1 if p > opt_threshold else 0 for p in y_prob_pred])

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')

    print(f"Precision: {precision} \n"
          f"Recall: {recall} \n"
          f"Fscore: {fscore} \n"
          f"Support: {support}")

    return (precision, recall, fscore, support)





# ------------------------------------------------------------------------------------------------------

def record_model(name, model, rate, featurelist, precision, recall, fscore):
    with open('pastmodels.txt', 'a') as file:
        file.write(f"{name}, {model}, {rate}\n")
        vars = ""
        for i in featurelist:
            vars = vars + ", " + i
        file.write(f"{vars}\n")
        scores = f"{round(precision, 3)}, {round(recall, 3)}, {round(fscore, 3)}\n"
        file.write(scores)
def display_dashboard():
    # Building dashboard UI
    # Setting hyperparameters, feature selection

    st.header("Dashboard")
    scale = st.checkbox("Scaling data")
    if (scale):
        st.write("Scaling data with standard scaler")
    else:
        st.write("Not scaling data")
    model = st.radio("Choose an algorithm: ",
                            ("Logistic", "Random Forest", "Decision Tree", "Support Vector Machine"))
    st.write("Selected", model, " as model")
    #select = st.selectbox("Choose evaluation metric: ", ['Precision', 'Recall', 'Precision-Recall'])
    #st.write("Selected ", select, " as metric")
    features = st.multiselect("Select features: ", df.columns[:-1].tolist())
    st.write("Selected ", len(features), " features")
    rate = st.slider("Select undersampling rate", min_value=0.0, max_value=1.0, step=0.05)
    st.write("Resampling rate set to ", rate)
    name = st.text_input("Enter model name / id",  "Name ...")
    st.write("Model to be named ", name)
    #slide = st.slider("Select level", 1, 10)
    #st.write("Currently at level: ", slide)

    if st.button("Submit"):
        st.write("Building model with existing hyperparameters...")
        precision, recall, fscore, support = train_logistic_model(name, scale, rate, features)
        st.write("Made model ", name, " with ", len(features),
                 " total features. \n Scaling was set to ", scale,
                 ".\n Precision: ", precision, "\n Recall: ", recall,
                 "\n Fscore: ", fscore, "\n Support: ", support)
        record_model(name, model, rate, features, precision, recall, fscore)

def display_logs():
    # Showing records of past models with data on evaluation metrics and hyperparameters

    st.header("Logs")
    with open('pastmodels.txt', 'r') as file:
        content = file.readlines()
        for line in content:
            st.write(line)
        print(content)

st.title("Credit Card Fraud Detection Model Development")
tab1, tab2 = st.tabs(["Dashboard", "Logs"])
with tab1:
    display_dashboard()
with tab2:
    display_logs()