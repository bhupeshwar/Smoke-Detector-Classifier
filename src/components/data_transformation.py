import sys
import os
import numpy as np
import pandas as pd 
from dataclasses import dataclass

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight

from src.exception import CustomException
from src.logger import logging
from src.utils import df_to_csv, handling_multicollinearity, handling_outliers, feature_scaling

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    validation_data_path: str = os.path.join('artifacts','validation_data.csv')
    scaled_data_path: str = os.path.join('artifacts', "scaled_data.csv")
    weights_path = os.path.join('artifacts', 'weights.npy')
    class_weights_path = os.path.join('artifacts', 'class_weights.npy')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self, data_path):

        logging.info("Started Data Transformation Component")
        try:
            df = pd.read_csv(data_path)

            # Initializing X and y for training purposes
            X = df.drop('Fire Alarm', axis=1)
            y = df['Fire Alarm']
            
            logging.info('Handling Multicollinearity')

            X = handling_multicollinearity(X)

            logging.info('Scaling the features')

            X_scaled = feature_scaling(X)

            df_to_csv(X_scaled, self.data_transformation_config.scaled_data_path)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            X_train = X_train[:-round(len(X_train)/10)]
            y_train = y_train[:-round(len(y_train)/10)]
            X_val = X_train[-round(len(X_train)/10):]
            y_val = y_train[-round(len(y_train)/10):]

            df_to_csv(X_train, self.data_transformation_config.train_data_path)
            df_to_csv(X_test, self.data_transformation_config.test_data_path)
            df_to_csv(X_val, self.data_transformation_config.validation_data_path)

            logging.info('Handling Outliers')

            train_weights, class_weights = handling_outliers(X_train, y_train)

            logging.info('Saving required weights')
            np.save(self.data_transformation_config.class_weights_path, class_weights)
            np.save(self.data_transformation_config.weights_path, train_weights)

            logging.info('Ended Data Transformation')

            return (
                y_train, y_test, y_val
            )
        
        except Exception as e:
            raise CustomException(e, sys)