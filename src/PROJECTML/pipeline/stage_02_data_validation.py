# then import the neccessary  libraries
from PROJECTML.config.configuration import ConfigurationManager
from PROJECTML.components.data_validation import DataValiadtion
from PROJECTML import logger

# here iam defined one class w.r.t name of the stage
STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline: # inside this DataValidationTrainingPipeline iam creating one main method
    def __init__(self):
        pass

    def main(self): # this the main method
        config = ConfigurationManager() # here i copy pasted these below 4 lines of code from data_validation.ipynb file try block
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()



# now iam calling the above code inside the below main fucntion

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline() # logging is started calling the class DataValidationTrainingPipeline
        obj.main() # here calling this main method from this class
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:  # if anything error in this code then we are telling this part of code to raise exception
        logger.exception(e)
        raise e