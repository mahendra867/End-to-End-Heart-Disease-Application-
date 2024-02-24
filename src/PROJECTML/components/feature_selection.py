import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from PROJECTML.entity.config_entity import FeatureSelectionConfig

# here i defined the component of DataTransformationConfig below
class FeatureSelection:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config

     

    def feature_selection(self):
        train_csv=pd.read_csv(self.config.train_data_path)
        test_csv=pd.read_csv(self.config.test_data_path)
        self.train_csv_features = train_csv[train_csv.columns.drop(['RestingBP','RestingECG'])]
        self.test_csv_features= test_csv[test_csv.columns.drop(['RestingBP','RestingECG'])]


                
# here i have defined the tarin_test_split below for performing the train_test_split
    def train_test_spliting(self):

        

        self.train_csv_features.to_csv(os.path.join(self.config.root_dir, "Modified_train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        self.test_csv_features.to_csv(os.path.join(self.config.root_dir, "Modified_test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(self.train_csv_features.shape) # this logs the information about that how many training and testing samples i have 
        logger.info(self.test_csv_features.shape)

        print(self.train_csv_features.shape)
        print(self.test_csv_features.shape)
