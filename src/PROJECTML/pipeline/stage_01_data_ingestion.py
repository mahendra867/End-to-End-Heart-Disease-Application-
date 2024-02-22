from PROJECTML.config.configuration import ConfigurationManager
from PROJECTML.components.data_ingestion import DataIngestion
from PROJECTML import logger 




STAGE_NAME = "Data Ingestion stage" # here iam naimg this stage as Data ingestion stage

# here iam creating a class  which is DataIngestionTrainingPipeline
class DataIngestionTrainingPipeline:
    def __init__(self): # and this consturctor file giving pass because it donot do anything here 
        pass

    def main(self): # here iam creating one method called main inside this just do copy past code which we written at the last part of the data_ingestion in data_ingestion.ipynb file 
        config = ConfigurationManager()  # this and below code steps belongs to pipeline
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


# now i need to call the above methode inside the below main methode basically it is telling that 
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") # my data ingestion is started
        obj = DataIngestionTrainingPipeline() # here iam initilizing this DataIngestionTrainingPipeline()
        obj.main() # here iam calling this main
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x") # then iam telling that this data ingestion stage is successfully running completed 
    except Exception as e:  # if there are any errors found this will get rise 
        logger.exception(e)
        raise e