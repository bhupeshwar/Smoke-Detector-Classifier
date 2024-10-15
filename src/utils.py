
"""
File: utils.py
Author: Bhupeshwar Pathania
Date: 02-10-2024
Description: Any functionalities used across the application can be initialized here

"""

import os
import sys
import pickle
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight


from src.exception import CustomException

# Helper functions for Data Transformation

def df_to_csv(df, path):
        try:
            # converting the df to csv
            df.to_csv(path, index=False, header=True)
        except Exception as e:
            raise CustomException(e, sys)
        
def handling_outliers(X, y):
    try:
        # Calculating class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

        # Assigning class weights to samples
        sample_weights = class_weights[y]

        # Isolation Forest with class weights
        iforest = IsolationForest(contamination=0.1, random_state=42)
        iforest.fit(X, sample_weight=sample_weights)

        # Outlier scores
        outlier_scores = iforest.decision_function(X)

        # Assign lower weights to outliers
        weights = np.where(outlier_scores < 0, 0.1, 1)
        weights *= sample_weights  # incorporate class weights

        return weights, class_weights
    
    except Exception as e:
        raise CustomException(e, sys)
    
def handling_multicollinearity(X):
    try:
        # As can be seen in the pairplot and heatmap from the juptyter notebook, there is linear relationship between few of the features (multicollinearity)
        # Handling multicollinearity with Variance Inflation Factor (VIF)
        vif = pd.DataFrame()
        vif['feature'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        print('Features with their respective VIFs')
        print(vif.sort_values(by='VIF', ascending=False))

        # Threshold for VIF
        max_vif = 5

        # Intializing a flag to check if any of the variables are exceeding the VIF threshold
        flag = True

        while flag:
            vif = pd.DataFrame()
            vif['feature'] = X.columns
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

            # Finding the variable with the highest VIF
            max_vif_feature = vif.loc[vif['VIF'].idxmax()]

            if max_vif_feature['VIF'] > max_vif:
                # Removing the variable from X 
                X = X.drop(max_vif_feature['feature'], axis=1)
                print(f'Variable with high VIF (removed): {max_vif_feature["feature"]} {max_vif_feature["VIF"]}')
            else:
                # If no variable exceeds the threshold, set the flag to False
                flag = False

        print('Final variables after handling multicollinearity', X.columns)

        return X
    
    except Exception as e:
        raise CustomException(e, sys)
        
def feature_scaling(X):
    try:
        scaler = StandardScaler()

        # Fit and transform on X_train
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled
    
    except Exception as e:
        raise CustomException(e, sys)
    
# Helper functions for Model Training

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e: 
        raise CustomException(e, sys)

# helper function to calculate the required metrics after the model training
def evaluate_model(true, predicted):
    try:
        return precision_score(true, predicted), recall_score(true, predicted), accuracy_score(true, predicted)
    except Exception as e: 
        raise CustomException(e, sys)

# helper functino to print the metrics
def print_metrics(precision, recall, accuracy):
    try:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'Accuracy: {accuracy}')
    except Exception as e: 
        raise CustomException(e, sys)