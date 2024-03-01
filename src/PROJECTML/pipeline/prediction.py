import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PROJECTML import logger
from PROJECTML.utils.common import load_bin



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
            self.model = load_bin("artifacts/model_trainer/model.pkl")
            print(self.model)
            logger.info("Model+preprocessing objects loaded successfully")
    
    
        def predict(self,data):
            
            prediction = self.model.predict(data)
            logger.info("Model predicted the Data")
            logger.info(f"Input data: {data}")
            logger.info(f"Predicted output: {prediction}")
            return prediction
