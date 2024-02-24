import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from PROJECTML.entity.config_entity import DataTransformationConfig
# here i defined the component of DataTransformationConfig below
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.mms = MinMaxScaler() # Move the MinMaxScaler initialization here
        self.ss = StandardScaler() # Move the StandardScaler initialization here

    def label_encoding(self):
        data = pd.read_csv(self.config.data_path)
        le = LabelEncoder()
        self.df1 = data.copy(deep=True)

        logger.info(f"Done with saving all features inside this {self.df1}")

        self.df1['Sex'] = le.fit_transform(self.df1['Sex'])
        self.df1['ChestPainType'] = le.fit_transform(self.df1['ChestPainType'])
        self.df1['RestingECG'] = le.fit_transform(self.df1['RestingECG'])
        self.df1['ExerciseAngina'] = le.fit_transform(self.df1['ExerciseAngina'])
        self.df1['ST_Slope'] = le.fit_transform(self.df1['ST_Slope'])

        logger.info(f"Done with label encoding for categorical features {self.df1}")

        # Save label encoders as joblib files
        label_encoders = {
            'Sex': le,
            'ChestPainType': le,
            'RestingECG': le,
            'ExerciseAngina': le,
            'ST_Slope': le
        }
        
        logger.info(f"Done with collecting the label_encoders feature values {label_encoders}")

        joblib.dump(label_encoders, os.path.join(self.config.root_dir, "label_encoders.joblib"))

        logger.info("Done with saving the label_encoders.joblib file")

    def data_scaling(self):
        self.df1['Oldpeak'] = self.mms.fit_transform(self.df1[['Oldpeak']])
        self.df1['Age'] = self.ss.fit_transform(self.df1[['Age']])
        self.df1['RestingBP'] = self.ss.fit_transform(self.df1[['RestingBP']])
        self.df1['Cholesterol'] = self.ss.fit_transform(self.df1[['Cholesterol']])
        self.df1['MaxHR'] = self.ss.fit_transform(self.df1[['MaxHR']])

        logger.info(f"Done with data_scalling for numerical features {self.df1}")

        return self.df1
        
    def train_test_spliting(self, scaled_data):
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(scaled_data, test_size=0.20,random_state=2)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Save preprocessed data as joblib files
        joblib.dump(train, os.path.join(self.config.root_dir, "train_data.joblib"))
        #joblib.dump(test, os.path.join(self.config.root_dir, "test_data.joblib"))

        logger.info("Done with saving the train_data.joblib file")

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        # Save scalers as joblib files
        scalers = {
            'Oldpeak': self.mms,
            'Age': self.ss,
            'RestingBP': self.ss,
            'Cholesterol': self.ss,
            'MaxHR': self.ss
        }

        logger.info(f"Done with collecting the scalers feature values {scalers}")

        joblib.dump(scalers, os.path.join(self.config.root_dir, "scalers.joblib"))

        logger.info("Done with saving the scalers.joblib file")