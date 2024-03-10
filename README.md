# End-to-End-Heart-Disease-Application with MLOps


# Project Title

A brief description of what this project does and who it's for


## Problem Statement
With a plethora of medical data available and the rise of Data Science, a host of startups are taking up the challenge of attempting to create indicators for the forseen diseases that might be contracted! Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Heart failure is a common event caused by CVDs. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help. In this way, we try to solve automate another problem that occurs in the nature with a view to counter it and focus on to the next problem with the help of AI techniques!

## Aim :
To classify / predict whether a patient is prone to heart failure depending on multiple attributes.
It is a binary classification with multiple numerical and categorical features.

## Dataset Attributes
- Age : age of the patient [years]
- Sex : sex of the patient [M: Male, F: Female]
- ChestPainType : chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP : resting blood pressure [mm Hg]
- Cholesterol : serum cholesterol [mm/dl]
- FastingBS : fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG : resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR : maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina : exercise-induced angina [Y: Yes, N: No]
- Oldpeak : oldpeak = ST [Numeric value measured in depression]
- ST_Slope : the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease : output class [1: heart disease, 0: Normal]



## Approach 


### Data Ingestion
I utilized Python libraries such as os, urllib.request, and zipfile for efficient data handling. The process starts by downloading the dataset from a specified URL using urllib.request.urlretrieve. If the file doesn't exist locally, it's downloaded. I logged relevant information using a custom logger. After downloading, the data is extracted using zipfile.ZipFile into a designated directory. Finally, I split the dataset into training and testing sets using train_test_split from scikit-learn and saved them as CSV files.

### Model Validation
For model validation, I created a DataValidation class. The validate_all_columns method checks whether all the columns in the dataset match those specified in the schema defined in the configuration file. If the validation fails, it logs the status as false; otherwise, it logs true. The status is written to a text file for reference.

### Data Transformation
The data transformation stage involves preprocessing the dataset for model training. I employed various techniques such as ordinal encoding for categorical features and feature scaling using MinMaxScaler and StandardScaler. These transformations were encapsulated within a pipeline for streamlined processing. After transformation, the modified datasets were saved as CSV files.

### Feature Selection
Feature selection aims to improve model efficiency by selecting the most relevant features. I dropped certain columns ('RestingBP' and 'RestingECG') from the dataset as part of feature selection. The modified datasets were then saved for further processing.

### Model Trainer
In the model training phase, I loaded the preprocessed data and utilized scikit-learn pipelines for seamless integration of preprocessing and model training. I used logistic regression as the classification algorithm. The model was trained on the training data and evaluated using accuracy, cross-validation score, and ROC-AUC score. The trained model was saved using joblib.dump.

### Model Evaluation
In the model evaluation stage, I loaded the test data and the trained model. I evaluated the model's performance using various metrics such as confusion matrix and classification report. Visualization techniques like heatmap were employed for better understanding. The evaluation results were logged for further analysis.


## Modular code WorkFlows

1. Update config.yam1
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the [main.py](http://main.py/)
9. Update the [app.py](http://app.py/)




# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/mahendra867/random_datasets/raw/main/Heart_csv.zip
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/mahendra867/ProjectML_with_MLFlow.mlflow \
MLFLOW_TRACKING_USERNAME=mahendra867 \
MLFLOW_TRACKING_PASSWORD=85969b2c9b582440861229562a757d53c3cbb020 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/mahendra867/ProjectML_with_MLFlow.mlflow

export MLFLOW_TRACKING_USERNAME=mahendra867

export MLFLOW_TRACKING_PASSWORD=85969b2c9b582440861229562a757d53c3cbb020

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

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

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 683781347713.dkr.ecr.us-east-1.amazonaws.com/projml

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model



