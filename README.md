# fraud-detection

## Overview:

Dataset can be found on Kaggle: 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
- Most features in data derived from PCA
- Feature names / descriptions are obfuscated for confidentiality
- Extreme class imbalance

## EDA

- Features are unknown and dataset has high dimensionality (~30 explanatory variables)
- Assume that not much information will be obtained from in-depth visualization / exploration of data
- Cannot apply domain knowledge on variables because of confidentiality
- Best to gain general idea of shape of dataset, distribution of classes, missing values count, and perform feature selection later on

## Model Training / Testing

- Ideas for current models include:
- Linear Regression as benchmark
- Logistic Regression / Probit Regression as benchmark
- Support Vector Machine
- Random Forest Classifier
- Decision Tree Classifier

- Feature selection necessary to prevent overfitting
- 