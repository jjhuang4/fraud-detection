import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

# Reading in dataset
df = pd.read_csv("creditcard.csv")

# ------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, precision_recall_fscore_support
from sklearn.metrics import roc_curve, RocCurveDisplay
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#features = list(df.columns[:-1])

def train(model, resample, scale, pca, features):
    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    steps = []
    if (resample == "Undersampling"):
        steps.append(('under', NearMiss(version=1, n_neighbors=3)))
    elif (resample == "Oversampling"):
        steps.append(('smote', SMOTE()))
        steps.append(('random', RandomUnderSampler()))
    if (scale or pca > 0):
        steps.append(('scale', StandardScaler()))
    if (pca > 0):
        steps.append(('pca', PCA(n_components=pca)))
    if (model == "Logistic"):
        steps.append(('clf', LogisticRegression(penalty='l2', class_weight='balanced', solver='newton-cholesky')))
    elif (model == "Random Forest"):
        steps.append(('clf', RandomForestClassifier(n_estimators=25, max_depth=10, class_weight='balanced', max_features='sqrt')))
    print(steps)
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)
    y_prob_pred = pipeline.predict_proba(X_test)[:, 1]

    return (y_prob_pred, y_test)
def evaluate(y_prob_pred, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold = thresholds[opt_idx]
    print("Optimal threshold: ", opt_threshold)
    RocCurveDisplay.from_predictions(y_test, y_prob_pred, color="red", plot_chance_level=True)

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pred)
    display = PrecisionRecallDisplay.from_predictions(y_test, y_prob_pred, name="Score")
    display.ax_.set_title("Precision-Recall Curve")

    y_pred = np.array([1 if p > opt_threshold else 0 for p in y_prob_pred])
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"Precision: {precision} \n"
          f"Recall: {recall} \n"
          f"Fscore: {fscore} \n"
          f"Support: {support}")

    return precision, recall, fscore, support


# ------------------------------------------------------------------------------------------------------

def record_model(name, model, resample, pca, featurelist, precision, recall, fscore):
    with open('pastmodels.txt', 'a') as file:
        file.write(f"{name}, {model}, {pca} principal components, {resample}\n")
        vars = ""
        for i in featurelist:
            vars = vars + i + ', '
        file.write(f"{vars}\n")
        scores = f"Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, Fscore: {round(fscore, 3)}\n"
        file.write(scores)
    with open('modelevals.csv', 'a') as file:
        csv.writer(file).writerow([name, model, resample, pca, len(featurelist),
                                  round(precision, 3), round(recall, 3), round(fscore, 3)])

def clear_logs():
    with open('pastmodels.txt', 'w') as file:
        pass
def clear_modelevals():
    with open('modelevals.csv', 'w') as file:
        csv.writer(file).writerow(['Name', 'Algorithm', 'Resample', 'PCA', 'Features',
                                   'Precision', 'Recall', 'Fscore'])
def display_dashboard_UI():
    # Building dashboard UI
    # Setting hyperparameters, feature selection

    st.header("Dashboard")
    resample = st.radio("Choose resampling method: ",
                        ("Oversampling", "Undersampling"))
    st.write("Selected ", resample, " as method")
    scale = st.checkbox("Scaling data")
    if (scale):
        st.write("Scaling data with standard scaler")
    else:
        st.write("Not scaling data")
    model = st.radio("Choose an algorithm: ",
                            ("Logistic", "Random Forest"))
    st.write("Selected ", model, " as model")
    #select = st.selectbox("Choose evaluation metric: ", ['Precision', 'Recall', 'Precision-Recall'])
    #st.write("Selected ", select, " as metric")
    features = st.multiselect("Select features: ", df.columns[:-1].tolist())
    st.write("Selected ", len(features), " features")
    pca = st.slider("Number of principle components (0 = no pca)", min_value=0, max_value=len(features)+1, step=1)
    #rate = st.slider("Select undersampling rate", min_value=0.0, max_value=1.0, step=0.05)
    #st.write("Resampling rate set to ", rate)
    name = st.text_input("Enter model name / id",  "Name ...")
    st.write("Model to be named ", name)
    #slide = st.slider("Select level", 1, 10)
    #st.write("Currently at level: ", slide)

    if st.button("Submit"):
        st.write("Building model with existing hyperparameters...")
        y_prob_pred, y_test = train(model, resample, scale, pca, features)
        precision, recall, fscore, support = evaluate(y_prob_pred, y_test)
        st.write("Made model ", name, " with ", len(features),
                 " total features and ", pca ," principle components. \n Scaling was set to ", scale,
                 ".\n Precision: ", precision, "\n Recall: ", recall,
                 "\n Fscore: ", fscore, "\n Support: ", support)
        record_model(name, model, resample, pca, features, precision, recall, fscore)

def display_logs_UI():
    # Showing records of past models with data on evaluation metrics and hyperparameters

    st.header("Logs")
    def show_models():
        with open('pastmodels.txt', 'r') as file:
            content = file.readlines()
            for line in content:
                st.write(line)
    show_models()
    st.write("\nWarning: clearing will delete all logs with no chance of undoing action.")
    st.write("Click twice to fully confirm deletion")
    if st.button("Clear model logs"):
        clear_logs()
        show_models()

def display_analysis_UI():
    try:
        df_evals = pd.read_csv('modelevals.csv')
        st.dataframe(df_evals)
    except Exception as e:
        st.write("Model records currently empty, add additional models")
        pass

    model_filter = st.radio("Algorithm used: ", ("Logistic", "Random Forest"))
    resample_filter = st.radio("Resampling method used: ", ("Undersampling", "Oversampling"))
    metrics = st.multiselect("Choose metrics: ", ['Precision', 'Recall', 'Fscore'])

    colors = []
    if len(metrics) == 0:
        metrics = ['Fscore']
    if len(metrics) == 1:
        colors = ["#FF5100"]
    elif len(metrics) == 2:
        colors = ["#FF5100", "#86FF45"]
    elif len(metrics) == 3:
        colors = ["#FF5100", "#86FF45", "#2667FF"]
    df_filtered = df_evals[(df_evals['Algorithm'] == model_filter) &
                           (df_evals['Resample'] == resample_filter)]
    st.dataframe(df_filtered)
    st.line_chart(df_filtered, x='PCA', y=metrics, color=colors)

    st.write("\nWarning: clearing will remove all model evaluation data with no chance of undoing action.")
    st.write("Click twice to fully confirm deletion")
    if st.button("Clear model data"):
        clear_modelevals()

st.title("Credit Card Fraud Detection Model Development")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Logs", "Analysis"])
with tab1:
    display_dashboard_UI()
with tab2:
    display_logs_UI()
with tab3:
    display_analysis_UI()