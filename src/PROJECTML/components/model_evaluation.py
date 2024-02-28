import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # here i have imported all the evaluation meterics for evaluating our model peformance
from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from PROJECTML.entity.config_entity import ModelEvaluationConfig
from src.PROJECTML.utils.common import load_bin




# here i have defined one ModelEvaluation class in which i have defined the ModelEvaluationConfig
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def model_evaluation(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=load_bin(self.config.model_path)
        test_x = test_data.drop([self.config.target_column], axis=1) # here iam dropping my targetcolumn in test_x
        test_y = test_data[[self.config.target_column]] # here iam keeping my target column
        # Confusion Matrix
        cm = confusion_matrix(test_y,model.predict(test_x))
        names = ['True Neg','False Pos','False Neg','True Pos']
        counts = [value for value in cm.flatten()]
        percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm,annot = labels,cmap = 'viridis',fmt ='')
    
    # Classification Report
        print(f"The classification report is :{classification_report(test_y,model.predict(test_x))}")



""" # here i have defiend the eval_metrics which it evaluates our model performance by calculating the rmse,mae,r2
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

# here i have defined one fucntion called log_into_mlflow which it will mlflow 
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path) # here iam loading the test dataset
        model = joblib.load(self.config.model_path) # here iam loading my trained model 

        test_x = test_data.drop([self.config.target_column], axis=1) # here iam dropping my targetcolumn in test_x
        test_y = test_data[[self.config.target_column]] # here iam keeping my target column


        mlflow.set_registry_uri(self.config.mlflow_uri) # here iam setting my tracking URI it is getting from configuration manager src because i want to everything inside a remote server 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x) # here iam predicting my testing data

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities) # here it is calculating my metrics of the predicted model
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2} # these scores are saving inside my artifacts of model_evaluation as a jason file
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params) # here iam logging with mlflow as log params,different log_mertics , and here iam giving all the parameters together which it will save inside ml flow

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

# here it will check whether we are doing inside the URI or in local directory , i already set the URL so it will save inside the URI 
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel") # here iam giving model ElasticnetModel
            else:
                mlflow.sklearn.log_model(model, "model") """