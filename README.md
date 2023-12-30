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

## Model Development Approach

Context of problem: credit card fraud. A fraudulent transaction mislabeled as benign is much more expensive and harmful to the credit company than a regular transaction mislabeled as fraudulent, which can be followed up on.

Because of highly imbalanced dataset, must consider implementing cost-sensitive learning.
- Important to stratify in train test split to preserve class distribution
- Implementing some form of resampling (under or over sampling of classes)
- Cost-sensitive modifications to algorithms
  - Alternatively, use costs as penalty for mis-classification (add to error)
  - Cost sensitive / class weighted logistic regression

Ideas for current models include:
- Logistic Regression / Probit Regression as benchmark
- Support Vector Machine
- Random Forest Classifier
- Decision Tree Classifier

Feature selection is necessary to prevent overfitting. Method of dealing with unbalanced datasets is under-sampling / over-sampling.

## Model Training Order

1. Define X explanatory variables and y target column
2. Perform stratified train_test_split on X and y data
3. Perform resampling on X_train and y_train
4. If scaling, scale X_train and X_test
5. Fit scaled and resampled data to the chosen algorithm
6. Evaluate the trained model with precision, recall, fscore and support