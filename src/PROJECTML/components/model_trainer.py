# these packages i need in order to create my Model Trainer components 
import pandas as pd
import matplotlib.pyplot as plt
import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import plot_roc_curve
import joblib # here iam saving the model because i want to save the data
from PROJECTML.entity.config_entity import ModelTrainerConfig


# now here iam defining a class called model trainer inside it will take ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config



    # here iam creating a methode which it will traine the model by using train and test dataset
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path) # here it is taking the paths of train and test dataset
        test_data = pd.read_csv(self.config.test_data_path)


        self.train_x = train_data.drop([self.config.target_column], axis=1)  # here iam dropping my target column in train_x
        self.test_x = test_data.drop([self.config.target_column], axis=1)  # here iam dropping my target column in test_X
        self.train_y = train_data[self.config.target_column]  # here iam keeping the target column in train_y
        self.test_y = test_data[self.config.target_column] # here iam keeping the target column in test_y


    def model(self):

        self.classifier_lr = LogisticRegression(random_state = 0, C=10, penalty= 'l2')
        self.classifier_lr.fit(self.train_x,self.train_y)
        prediction = self.classifier_lr.predict(self.test_x)
        cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
        print("Accuracy : ",'{0:.2%}'.format(accuracy_score(self.test_y,prediction)))
        print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(self.classifier_lr,self.train_x,self.train_y,cv = cv,scoring = 'roc_auc').mean()))
        print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(self.test_y,prediction)))


        
        # lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42) # here i have created my Elastic model which it takes the alpha,l1_ratio, random state values 
        # lr.fit(train_x, train_y) # here i have initiated the model training

        joblib.dump(self.classifier_lr, os.path.join(self.config.root_dir, self.config.model_name)) # here are training my model iam just saving inside the folder Model_trainer which it will get create inside the artifacts