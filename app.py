from flask import Flask, render_template, request # here i have imported some necesary libraries which helps us create User interface  
import os 
import numpy as np
import pandas as pd
from PROJECTML.pipeline.prediction import PredictionPipeline # here iam importing the predicition pipeline which it do predicts on the uploaded data by user and it return the prediction value


# here iam initilize my flask with default route , so whenver we execute the app.py we get one host and port number if we click on it it will open oir index.html file which is present inside the template 
app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

# here iam defining 1 training route because user can train from here 
@app.route('/train',methods=['GET'])  # route to train the pipeline , here i have given /train it means we are running the main.py 
def training(): # which i passed main.py inside the training() , by using os.system we can execute any python command 
    os.system("python main.py") # so this command  python main.py that we need to execute in our terminal so instead of writing the command in terminal i have given insdie the os.system it will execute the python command , so it will start the main.py it means it starts the training pipeline  
    return "Training Successful!" 

# here iam creating the prediction route which it helps us to take the new values to the independent columns then our saved best model do prediction and it returns the predicted value
@app.route('/predict',methods=['POST','GET']) 
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Age = int(request.form['Age'])
            Sex = request.form['Sex']
            ChestPainType = request.form['ChestPainType']
            Cholesterol = int(request.form['Cholesterol'])
            FastingBS = int(request.form['FastingBS'])
            MaxHR = int(request.form['MaxHR'])
            ExerciseAngina = request.form['ExerciseAngina']
            Oldpeak = float(request.form['Oldpeak'])
            ST_Slope = request.form['ST_Slope']
            
        

       
         
            data = [Age,Sex,ChestPainType,Cholesterol,FastingBS,MaxHR,ExerciseAngina,Oldpeak,ST_Slope] # here iam taking the passed independent new values by user in the form of list
            data = np.array(data).reshape(1, 9) # here iam converting to numpy array of that data , make sure that change the feature values according to problem statement here in this dataset iam going to take 9 features by user that's y i mention the 9 as feature value reshpaing
            
            obj = PredictionPipeline() # here iam initilizing the PredictionPipeline() and on the top of it calling the predic
            predict = obj.predict(data)
            print(predict)
            return render_template('results.html', prediction = str(predict)) # here iam returning the prediction

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True) 
	app.run(host="0.0.0.0", port = 8081) # here host , port values are below as follows