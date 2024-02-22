from PROJECTML.entity.config_entity import DataIngestionConfig 
from PROJECTML.entity.config_entity import DataValidationConfig
from PROJECTML.entity.config_entity import DataTransformationConfig
from PROJECTML.entity.config_entity import FeatureSelectionConfig
from PROJECTML.entity.config_entity import ModelTrainerConfig
from PROJECTML.entity.config_entity import ModelEvaluationConfig
from PROJECTML.constants import * # here iam importing everthing which is present in the constants->__init__.py file into inside the data_ingestion.ipynb
from PROJECTML.utils.common import read_yaml, create_directories # here iam importing the read_yaml, create_directories which are presenting inside the utils,common files into PROJECTML in which the file is data_ingestion.ipynb

class ConfigurationManager:  # here iam creating class called ConfigurationManager
    def __init__( # inisde this class iam reading all the yaml files which iam calling it from constants->__init__.py file and iam mentioning inside the class varaiable 
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath) # and here iam giving read_yaml path here and iam giving the path after that then it will return all the configuration in the variable
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

# now i will create artifacts root in the side of the vscode project one of the path and the below i will define the data ingestion cofiguration function
    # the above one  entity which inside 4 variables needs to return by this below fucntion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir]) # here iam creating the root directory, and iam reading the config from the configurationManager class and iam going to access all the data ingestion from the config.yaml file 

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,  # that how iam accessing all the things like root_dir,source_url and etc from config.yaml file and finally this fucntion do return all this variables 
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    # then i will prepare my get_data_validation_config , so inside this iam returning all of the varaibles that i have deifined inside my entity 
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation  # after reading by config iam returning the root_dir,status_file etc one by one
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig( # the above entity code is return type , and the below varaibles are getting return after reading by config varaible 
            root_dir=config.root_dir, 
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    # # only this part get changes in every step, only defining the get_data_transformation_config get changes according to which step we are performing like 01_data_ingestion,02_data_validation  
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,  # here iam returning these 2 varaibles by using this code 
            data_path=config.data_path,
        )

        return data_transformation_config
    

    # only this part get changes in every step, only defining the get_data_transformation_config get changes according to which step we are performing like 01_data_ingestion,02_data_validation
    def get_feature_selection_config(self) -> FeatureSelectionConfig:
        config = self.config.feature_selection

        create_directories([config.root_dir])

        feature_selection_config = FeatureSelectionConfig(
            root_dir=config.root_dir,  # here iam returning these 2 varaibles by using this code 
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
        )

        return feature_selection_config
    
    # this is part of code for the Model trainerConfig which helps us to return the configuration
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer   # here iam reading the schema, params 
        #params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            target_column = schema.name # here from schema iam taking the name which i will return through target_column
            
        )

        return model_trainer_config # here iam returning all variables from the configuration
    

    # only this part of code get changes her actually iam defining a get_model_evaluation_config
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation# here our model ElasticNet reading all params , Target_column of schema 
        #params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            # all_params=params, # here iam saving all the parameters which i do experment while getting to know at which parameters our trained model which is ElastcNet is doing well
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            # mlflow_uri="https://dagshub.com/mahendra867/ProjectML_with_MLFlow.mlflow", # here i have setted the mlflow URI
           
        )

        return model_evaluation_config # so which ever the varaibles we initilized inside the entity those are all getting returned
