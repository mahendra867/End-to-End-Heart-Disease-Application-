import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

# here iam creating 1 class inside the class iam going to load the best model that we got by comparing the parameters through ml flow UI
class PredictionPipeline:
    def __init__(self): # here i have created one construtor
        self.model = joblib.load(Path('artifacts\model_trainer\model.joblib')) # here iam using joblib for loading the model and i have mention the path which got saved inside this path artifacts/model_trainer/model.joblib

    # here iam defining 1 methode which this will take data from the user and our model do the predicition 
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction # here it will return the predicted value
    
