from flask import Flask, render_template, request # here i have imported some necesary libraries which helps us create User interface  
import os 
import numpy as np
import pandas as pd
from PROJECTML.pipeline.prediction import PredictionPipeline,CustomData # here iam importing the predicition pipeline which it do predicts on the uploaded data by user and it return the prediction value
from PROJECTML import logger


app = Flask(__name__)



@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) 
def predict_datapoint():
    if request.method=="GET":  # here by using this GET method to the 2nd url which is /predict client(user) sending the requesting to the backend code which is this source code ,by taking this form.html code content it provides the user interface so the client or user pass the data to the backend server so as soons as i click on submit button user or client sends request or submites  to the backend source code which the post Method perform this 
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


        dataframe=data.get_data_as_dataframe()  # here iam calling the get_data_as_dataframe() method with  data which is obejct for the class customdata and we are storing the gathered data of individual feature which is in the form of dataframe  gets stored  in final_data variable 

        logger.info('initiated prediction')
        predict_pipeline=PredictionPipeline() # now here i have created object for the predictionpipeline class to assign the gathered data in the form of dataframe to the best model which i have defined inisde this PredictPipeline() class 
        data = np.array(dataframe).reshape(1, 9)
        prediction=predict_pipeline.predict(data)
        # result=round(pred[0],2)  # here in the result variable we have prediction value with array datatype so iam taking the intial value by just rounding off to 2 points
        logger.info('made prediction and returning to results.html')
        return render_template("results.html",final_result=str(prediction)) # now iam going to create another html file which is result.html to that iam passing this result which consist of prediction value 
logger.info('done with prediction')

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8081)
