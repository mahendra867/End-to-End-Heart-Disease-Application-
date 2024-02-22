import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from PROJECTML.entity.config_entity import DataTransformationConfig

# here i defined the component of DataTransformationConfig below
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def label_encoding(self):
        data=pd.read_csv(self.config.data_path)
        le = LabelEncoder()
        self.df1 = data.copy(deep = True)

        self.df1['Sex'] = le.fit_transform(self.df1['Sex'])
        self.df1['ChestPainType'] = le.fit_transform(self.df1['ChestPainType'])
        self.df1['RestingECG'] = le.fit_transform(self.df1['RestingECG'])
        self.df1['ExerciseAngina'] = le.fit_transform(self.df1['ExerciseAngina'])
        self.df1['ST_Slope'] = le.fit_transform(self.df1['ST_Slope'])

    
    def data_scaling(self):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization

        self.df1['Oldpeak'] = mms.fit_transform( self.df1[['Oldpeak']])
        self.df1['Age'] = ss.fit_transform( self.df1[['Age']])
        self.df1['RestingBP'] = ss.fit_transform( self.df1[['RestingBP']])
        self.df1['Cholesterol'] = ss.fit_transform( self.df1[['Cholesterol']])
        self.df1['MaxHR'] = ss.fit_transform( self.df1[['MaxHR']])

        
        return self.df1

        
        
# here i have defined the tarin_test_split below for performing the train_test_split
    def train_test_spliting(self,scalled_data):


        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(scalled_data,test_size=0.20) # this line splits the data into train_test_split


        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape) # this logs the information about that how many training and testing samples i have 
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
