import os
import sys
from dataclasses import dataclass

# Add the project's root directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, print_metrics

import pandas as pd
import numpy as np

# Modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

@dataclass
class TrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")

class Trainer:
    def __init__(self):
        self.trainer_config = TrainerConfig()
    
    def initiate_model_trainer(self, y_train, y_test, y_val):
        logging.info('Started Model Training Component')
        try:
            logging.info('Loading required files')

            X_train = pd.read_csv('artifacts/train.csv')
            X_test = pd.read_csv('artifacts/test.csv')
            X_val = pd.read_csv('artifacts/validation_data.csv')
            train_weights = np.load('artifacts/weights.npy')
            class_weights = np.load('artifacts/class_weights.npy')

            logging.info('Initializing classifiers')
            
            # Model training
            models = {
                "Decision Tree Classifier": DecisionTreeClassifier(ccp_alpha=0.01, class_weight='balanced'),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=2, class_weight='balanced'),
                "XGBClassifier": XGBClassifier(colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=class_weights[1]),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False, l2_leaf_reg=0.1, class_weights=class_weights),
                "LightGBM Classifier": LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=-1, learning_rate=0.05, class_weight='balanced',silent=True,verbose=-1),
                "AdaBoost Classifier": AdaBoostClassifier(estimator=DecisionTreeClassifier(ccp_alpha=0.01), n_estimators=100, learning_rate=0.1, algorithm='SAMME')
            }

            logging.info('Training the classifiers')
            for name, clf in models.items():
                print(f'Classifier: {name}')
                
                # Training the classifier
                clf.fit(X_train, y_train, sample_weight=train_weights)

                # Predictions
                # training
                y_pred_train = clf.predict(X_train)

                # testing
                y_pred_test = clf.predict(X_test)

                # validation
                y_pred_val = clf.predict(X_val)

                # Evaluation
                # training
                precision1, recall1, accuracy1 = evaluate_model(y_train, y_pred_train)

                # testing
                precision2, recall2, accuracy2 = evaluate_model(y_test, y_pred_test)

                # validation
                precision3, recall3, accuracy3 = evaluate_model(y_val, y_pred_val)

                print('-'*50)

                # Scores
                print(f'Performance Metrics of {name}')
                print('Training:')
                print_metrics(precision1, recall1, accuracy1)

                print('-'*25)

                print('Testing:')
                print_metrics(precision2, recall2, accuracy2)

                print('-'*25)

                print('Validation:')
                print_metrics(precision3, recall3, accuracy3)
                
                print('='*50)
                print('\n')
            
            logging.info('Choosing the best model')
            # As shown in the notebook, any classifier from above is considered good. So I will store the weights of randomly chosen classifier and use it later for prediction
            # LightGBM
            best_model = list(models.values())[4]

            save_object(
                file_path = self.trainer_config.model_file_path,
                obj = best_model
            )

            logging.info('Saved the model successfully')
            
        except Exception as e:
            raise CustomException(e, sys)