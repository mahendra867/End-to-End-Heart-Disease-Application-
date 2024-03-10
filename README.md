# End-to-End-Heart-Disease-Application with MLOps

## Problem Statement :
With a plethora of medical data available and the rise of Data Science, a host of startups are taking up the challenge of attempting to create indicators for the forseen diseases that might be contracted! Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Heart failure is a common event caused by CVDs. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help. In this way, we try to solve automate another problem that occurs in the nature with a view to counter it and focus on to the next problem with the help of AI techniques!

Aim :
To classify / predict whether a patient is prone to heart failure depending on multiple attributes.
It is a binary classification with multiple numerical and categorical features.
Dataset Attributes
Age : age of the patient [years]
Sex : sex of the patient [M: Male, F: Female]
ChestPainType : chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP : resting blood pressure [mm Hg]
Cholesterol : serum cholesterol [mm/dl]
FastingBS : fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG : resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR : maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina : exercise-induced angina [Y: Yes, N: No]
Oldpeak : oldpeak = ST [Numeric value measured in depression]
ST_Slope : the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease : output class [1: heart disease, 0: Normal]


## WorkFlows
Update config.yam1
Update schema.yaml
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the app.py


## Approach

Data Ingestion:
In the data ingestion stage, I, as the developer, first ensure that the necessary libraries are imported for component updates. The DataIngestion class is then defined to handle the ingestion process based on the provided configuration. Within this class, two main methods are implemented:

download_file: This method downloads the data from a specified URL using the urllib library and saves it locally. It checks if the file already exists and logs its size accordingly.

extract_zip_file: Here, the downloaded zip file is extracted into a designated directory using the zipfile library. This ensures that the data is ready for further processing.

Data Validation:
In the data validation stage, I create the DataValidation component to ensure the integrity and completeness of the ingested data. The class DataValidation contains the following key method:

validate_all_columns: This method reads the unzipped data and compares its columns against a predefined schema. If all columns match the schema, it returns a validation status of True; otherwise, it returns False. The status is then written into a text file for reference.

Data Transformation:
In this stage, I focus on transforming the data to make it suitable for modeling. The DataTransformation component, primarily featuring the train_test_spliting method, handles this process.

train_test_spliting: Using train_test_split from sklearn, the method splits the data into training and testing sets. It saves these splits as CSV files for future use.

Model Training:
Moving forward, in the model training stage, I develop the ModelTrainer component to train a predictive model using the prepared data. The class ModelTrainer incorporates:

train: This method reads the training and testing data, separates the features and target variable, and initializes an ElasticNet model. The model is trained on the training data and saved using joblib.

Model Evaluation:
Lastly, in the model evaluation stage, I assess the performance of the trained model. The ModelEvaluation component, with the method log_into_mlflow, handles this evaluation:

log_into_mlflow: Using mlflow, the method loads the test dataset and the trained model, makes predictions, and calculates evaluation metrics such as RMSE, MAE, and R2. These metrics are logged and saved, and the trained model is registered with mlflow for further tracking and deployment.

## How to run?

STEPS:
Clone the repository

https://github.com/mahendra867/random_datasets/raw/main/winequality-data.zip
STEP 01- Create a conda environment after opening the repository
conda create -n mlproj python=3.8 -y
conda activate mlproj
STEP 02- install the requirements
pip install -r requirements.txt
# Finally run the following command
python app.py
Now,

open up you local host and port
MLflow
Documentation

cmd
mlflow ui
dagshub
dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/mahendra867/ProjectML_with_MLFlow.mlflow
MLFLOW_TRACKING_USERNAME=mahendra867
MLFLOW_TRACKING_PASSWORD=85969b2c9b582440861229562a757d53c3cbb020
python script.py

Run this to export as env variables:

export MLFLOW_TRACKING_URI=https://dagshub.com/mahendra867/ProjectML_with_MLFlow.mlflow

export MLFLOW_TRACKING_USERNAME=mahendra867

export MLFLOW_TRACKING_PASSWORD=85969b2c9b582440861229562a757d53c3cbb020
AWS-CICD-Deployment-with-Github-Actions
1. Login to AWS console.
2. Create IAM user for deployment
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
3. Create ECR repo to store/save docker image
- Save the URI: 683781347713.dkr.ecr.us-east-1.amazonaws.com/projml
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one
7. Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app
About MLflow
MLflow

Its Production Grade
Trace all of your expriements
Logging & tagging your model

ECR reposiotry URI = 683781347713.dkr.ecr.us-east-1.amazonaws.com/heartdisease_application_image_repository
