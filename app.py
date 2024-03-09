from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from PROJECTML.pipeline.prediction import PredictionPipeline,CustomData
from PROJECTML import logger

app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) 
def predict_datapoint():
    if request.method=="GET":
        return render_template("index.html")  
    else:
        data = CustomData(
                            Age=int(request.form.get("Age")),
                            Sex=request.form.get("Sex"),
                            ChestPainType=request.form.get("ChestPainType"),
                            Cholesterol=int(request.form.get("Cholesterol")),
                            FastingBS=int(request.form.get("FastingBS")),
                            MaxHR=int(request.form.get("MaxHR")),
                            ExerciseAngina=request.form.get("ExerciseAngina"),
                            Oldpeak=float(request.form.get("Oldpeak")),
                            ST_Slope=request.form.get("ST_Slope")
        )

        dataframe=data.get_data_as_dataframe()
        logger.info('initiated prediction')
        predict_pipeline=PredictionPipeline()
        data = np.array(dataframe).reshape(1, 9)
        prediction=predict_pipeline.predict(data)
        final_result = str(int(prediction[0]))  # Convert numpy array to integer first, then to string
        logger.info('made prediction and returning to results.html')
        return render_template("results.html", final_result=final_result)

logger.info('done with prediction')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9095)
