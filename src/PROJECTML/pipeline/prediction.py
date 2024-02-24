import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PROJECTML import logger
from PROJECTML.utils.common import load_object
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


class CustomData:
    def __init__(
        self,
        Age: int,
        Sex: object,
        ChestPainType: object,
        Cholesterol: int,
        FastingBS: int,
        MaxHR: int,
        ExerciseAngina: object,
        Oldpeak: float,
        ST_Slope: object,
    ):

        self.Age = Age
        self.Sex = Sex
        self.ChestPainType = ChestPainType
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.MaxHR = MaxHR
        self.ExerciseAngina = ExerciseAngina
        self.Oldpeak = Oldpeak
        self.ST_Slope = ST_Slope
        logger.info("features values got stored inside the CustomData class")

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = { # here i passing  the features (which has values which passed by user) in sequence of how the original dataset has features sequence
                "Age": [self.Age],
                "Sex": [self.Sex],
                "ChestPainType": [self.ChestPainType],
                "Cholesterol": [self.Cholesterol],
                "FastingBS": [self.FastingBS],
                "MaxHR": [self.MaxHR],
                "ExerciseAngina": [self.ExerciseAngina],
                "Oldpeak": [self.Oldpeak],
                "ST_Slope": [self.ST_Slope]
            }
            self.df = pd.DataFrame(custom_data_input_dict)
            logger.info("Dataframe Gathered")
            logger.info(f"Dataframe gathered values are {self.df}")
            return self.df
            

        except Exception as e:
            logger.info("Exception Occurred in prediction pipeline")
            raise e


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts\model_trainer\model.joblib"))
        self.label_encoder = joblib.load(Path("artifacts\data_transformation\label_encoders.joblib"))
        self.scaler = joblib.load(Path("artifacts\data_transformation\scalers.joblib"))
        logger.info("Model and preprocessing objects loaded successfully")
        
        
    def preprocess_data(self, dataframe):
        df = dataframe
        logger.info(f"For Prediction pipeline iam taking this Dataframe which are values passed by user  {df}")
        # Apply label encoding to categorical features
        # From here iam getting error data is not getting transformation by joblib folders
        df["Sex"] = self.label_encoder["Sex"].transform(df["Sex"])
        df["ChestPainType"] = self.label_encoder["ChestPainType"].transform(df["ChestPainType"])
        df["ExerciseAngina"] = self.label_encoder["ExerciseAngina"].transform(df["ExerciseAngina"])
        df["FastingBS"] = self.label_encoder["FastingBS"].transform(df["FastingBS"])
        df["ST_Slope"] = self.label_encoder["ST_Slope"].transform(df["ST_Slope"])
        logger.info(f"done with label encoding and this the result for the categorical features after label encoding {df}")
        
        # Apply data scaling to numerical features
        df[["Age", "Cholesterol", "MaxHR", "Oldpeak"]] = self.scaler.transform(df[["Age", "Cholesterol", "MaxHR", "Oldpeak"]])
        logger.info(f"done with feature scalling  and this the result for the numerical features after feature scalling for numerical features  {df}")
        
        return df

    def predict(self, preprocessed_data):
        prediction = self.model.predict(preprocessed_data)
        logger.info("Model predicted the Data")
        logger.info(f"Input data: {preprocessed_data}")
        logger.info(f"Predicted output: {prediction}")
        return prediction
