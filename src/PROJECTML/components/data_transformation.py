import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from PROJECTML.entity.config_entity import DataTransformationConfig
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import joblib
from PROJECTML.utils.common import load_object




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.mms = MinMaxScaler() # Move the MinMaxScaler initialization here
        self.ss = StandardScaler() # Move the StandardScaler initialization here

    def pipeline_creation(self):
    
        train=pd.read_csv("artifacts\\data_ingestion\\train.csv")
        test=pd.read_csv("artifacts\\data_ingestion\\test.csv")

        x_train=train.drop(columns=['HeartDisease'])
        y_train=train['HeartDisease']
        x_test=test.drop(columns=['HeartDisease'])
        y_test=test['HeartDisease']

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)


        x_train_index = x_train.columns.tolist()  # Get the column names as a list
        y_train_index = y_train.name  # Get the name of the target column

        print("Index values of x_train:", x_train_index)
        print("Index value of y_train:", y_train_index)


        logger.info("<<<<<<<---------------------Started Data_transformation Pipeline ------------------------>>>>>>>>>>>>>>>>")

        trf1 = ColumnTransformer(
        [('ordinal_encode', OrdinalEncoder(), [1, 2, 6, 8])],  # Indices of categorical columns
            remainder='passthrough'
        )

        logger.info("<<<<<<<---------------------Started feature scalling Pipeline ------------------------>>>>>>>>>>>>>>>>")

        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        trf2 = ColumnTransformer([
            ('minmax_scale', minmax_scaler, [7]),
            ('standard_scale', standard_scaler, [0,3,5])],remainder='passthrough')


        print(trf2)

        logger.info("<<<<<<<---------------------Started creating Model Pipeline ------------------------>>>>>>>>>>>>>>>>")

        trf3=LogisticRegression(random_state=0,C=10,penalty='l2',max_iter=2000)
        print(trf3)

        logger.info("<<<<<<<---------------------Started Combining all Pipeline into one pipeline------------------------>>>>>>>>>>>>>>>>")
        pipe=Pipeline([
            ('trf1',trf1),
            ('trf2',trf2),
            ('trf3',trf3)
        ])  

        logger.info("<<<<<<<---------------------Making all Pipeline into one pipeline------------------------>>>>>>>>>>>>>>>>")
        pipe = make_pipeline(trf1, trf2, trf3)
        print(pipe)

        
       
        logger.info(f"Done with ordinal encoding for categorical features on train and test")

        
                
        
        logger.info("Done with data_scalling for numerical features on train and test data")

        logger.info("Returning the pipeline")

        logger.info("<<<<<<<------------------------------------------------------->>>>>>>>>>>>>>>>")

        train.to_csv(os.path.join(self.config.root_dir, "Transformed_train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        test.to_csv(os.path.join(self.config.root_dir, "Transformed_test.csv"),index = False)

        return pipe
