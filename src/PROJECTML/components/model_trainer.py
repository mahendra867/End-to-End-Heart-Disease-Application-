# these packages i need in order to create my Model Trainer components 
import pandas as pd
import matplotlib.pyplot as plt
import os
from PROJECTML import logger
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import plot_roc_curve
import joblib # here iam saving the model because i want to save the data
from src.PROJECTML.config.configuration import ConfigurationManager
from src.PROJECTML.components.data_transformation import DataTransformation
from PROJECTML.entity.config_entity import ModelTrainerConfig

import pickle

# now here iam defining a class called model trainer inside it will take ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    # here iam creating a methode which it will traine the model by using train and test dataset
    def train(self):
        self.train_data = pd.read_csv(self.config.train_data_path) # here it is taking the paths of train and test dataset
        self.test_data = pd.read_csv(self.config.test_data_path)

        self.x_train=self.train_data.drop(columns=['HeartDisease'])
        self.y_train=self.train_data['HeartDisease']
        self.x_test=self.test_data.drop(columns=['HeartDisease'])
        self.y_test=self.test_data['HeartDisease']
        
        


    def model(self):

        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        pipeline_object=data_transformation.pipeline_creation()
        print(f"got  the pipeline_object {pipeline_object}")

        pipe = pipeline_object
        pipe.fit(self.x_train, self.y_train)
        prediction = pipe.predict(self.x_test)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        print("Accuracy : ", '{0:.2%}'.format(accuracy_score(self.y_test, prediction)))
        print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score(pipe, self.x_train, self.y_train, cv=cv, scoring='roc_auc').mean()))
        print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc_score(self.y_test, prediction)))

    



        
        # lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42) # here i have created my Elastic model which it takes the alpha,l1_ratio, random state values 
        # lr.fit(train_x, train_y) # here i have initiated the model training

        #pickle.dump(pipe,os.path.join(self.config.root_dir,self.config.model_name),open('model.pkl','wb')) # here are training my model iam just saving inside the folder Model_trainer which it will get create inside the artifacts

        joblib.dump(pipe, os.path.join(self.config.root_dir, self.config.model_name))

        #with open(os.path.join(self.config.root_dir, self.config.model_name), 'wb') as model_pkl_file:
         #   pickle.dump(pipe, model_pkl_file)
