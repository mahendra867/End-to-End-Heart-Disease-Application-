# This is called the entity 
from dataclasses import dataclass # here i imported the dataclass from the dataclasses
from pathlib import Path  # here i imported path from pathlib

# here entity means DataIngestionConfig which it returns all the variables like root_dir,source_URL  and etc 
@dataclass(frozen=True) # here i declared the dataclass decorator
class DataIngestionConfig:  # here i have created a class and named as DataIngestionConfig ,and it is not a python class because we need to declare the self to the variables if it is a python class, it is data class  and whenever i define the configuration fucntion , this class should my return function , the below are the varaible it do return 
    root_dir: Path    # these are variable which i have declared inside the class 
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


# lets prepare the entity here , and if i open the config.yaml file where i can find the root_dir,status_file,unzip_data the same i do find in the class iside the DataValiadationConfig
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict # here all_Schema just read all the data and install inside the all_schema varaible as a dictionary formate



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path  # these are variables which are present inside the config.yaml file data_transformation code part and here iam mentioning inside the entity of the class
    data_path: Path


@dataclass(frozen=True)
class FeatureSelectionConfig:
    root_dir: Path  # these are variables which are present inside the config.yaml file data_transformation code part and here iam mentioning inside the entity of the class
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str  # this target column is present inside the Schema.yaml file which it tells us the quality of the Wine based on the value it returns
    


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path  # this root_dir,test_data_path,model_path,metric_file_name are the variables takes from the model_evaluation configuration code part of config.yaml file 
    test_data_path: Path
    model_path: Path
    #all_params: dict # these all params is available inside the params.yaml file which it reads all the parameters 
    metric_file_name: Path
    target_column: str # this target column takes from schema.yaml file 
    #mlflow_uri: str # this mlflow uri iagain write it down once again inside mlflow URI section