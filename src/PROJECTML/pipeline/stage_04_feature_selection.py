from PROJECTML.config.configuration import ConfigurationManager
from PROJECTML.components.feature_selection import FeatureSelection
from PROJECTML import logger 




STAGE_NAME = "Feature Selection stage" # here iam naimg this stage as Data ingestion stage

# here iam creating a class  which is DataIngestionTrainingPipeline
class FeatureSelectionPipeline:
    def __init__(self): # and this consturctor file giving pass because it donot do anything here 
        pass

    def main(self): # here iam creating one method called main inside this just do copy past code which we written at the last part of the data_ingestion in data_ingestion.ipynb file 
        config = ConfigurationManager() # here iam initlizing my ConfigurationManager
        feature_selection_config = config.get_feature_selection_config() # and here iam getting my get_data_transformation_config()
        feature_selection = FeatureSelection(config=feature_selection_config) # here iam passing my data_transformation_config it means iam calling this data_transformation_config
        feature_selection.feature_selection()
        feature_selection.train_test_spliting() # here performing the train_test_split()


# now i need to call the above methode inside the below main methode basically it is telling that 
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") # my data ingestion is started
        obj = FeatureSelectionPipeline() # here iam initilizing this DataIngestionTrainingPipeline()
        obj.main() # here iam calling this main
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x") # then iam telling that this data ingestion stage is successfully running completed 
    except Exception as e:  # if there are any errors found this will get rise 
        logger.exception(e)
        raise e