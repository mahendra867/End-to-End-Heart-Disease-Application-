# these are libraries i need for to uodate the components 
import os
import urllib.request as request # so i use the request to download the data from the URL
import zipfile # here iam using the Zipfile to transform the data 
from PROJECTML import logger # here logger is used to logger the data 
from PROJECTML.utils.common import get_size # here i used the getsize is used to get to know the file size
from pathlib import Path
from PROJECTML.entity.config_entity import DataIngestionConfig 


# Now iam going to define one class which is DataIngestion from that class which it will take the DataIngestionConfig because from this dataingestionConfig only it will get to know the path 
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

# now i will define one method which it is responsible for dowmloading the data 
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL, # it will download the dta from this URL
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}") # if the data file is already exist it will print the message like that data file is already exit


# now iam going to another method called ExtractZipfile
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref: # here it will take the local_data_file path which is present in the config.yaml  local_data_file: artifacts/data_ingestion/data.zip and it will unzip the folder to this data_ingestion 
            zip_ref.extractall(unzip_path)