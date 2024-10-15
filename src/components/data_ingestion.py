import os
import sys
import pandas as pd


# Add the project's root directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Trainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Started Data Ingestion Component')
        try:
            df = pd.read_csv('data/smoke.csv')

            logging.info('Read the required data as pandas dataframe')

            logging.info('Data Cleaning Begins')

            logging.info('Dropping unncessary columns')

            # Dropping unnecessary columns
            df = df.drop(['Unnamed: 0', 'UTC', 'CNT'], axis=1)

            logging.info('Renaming column names')

            df = df.rename(columns={
                'Temperature[C]': 'Temperature',
                'Humidity[%]': 'Humidity',
                'TVOC[ppb]': 'TVOC',
                'eCO2[ppm]': 'eCO2',
                'Pressure[hPa]': 'Pressure'
            })

            logging.info('Checking for missing values')

            # Checking for missing values
            print(df.isnull().sum())

            logging.info('Checking for duplicate records')

            # Checking for any duplicacte records
            print('Duplicates:', df.duplicated().sum())

            logging.info('Checking the distribution of target variable')

            # Checking the distribution of target variable
            print(df['Fire Alarm'].value_counts())
            print(round(df['Fire Alarm'].value_counts(normalize=True)*100,2))

            logging.info('Data Cleaning Ends')

            # creating a file to store raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # converting the df to csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data Ingestion is completed')

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    di = DataIngestion()
    data_path = di.initiate_data_ingestion()

    dt = DataTransformation()
    y_train, y_test, y_val = dt.initiate_data_transformation(data_path)

    mt = Trainer()
    mt.initiate_model_trainer(y_train, y_test, y_val)