# Now iam going to create the components 
# First i will import some things 
import os
from PROJECTML import logger
from PROJECTML.entity.config_entity import DataValidationConfig
import pandas as pd



# Now i will create DataValidation class 

class DataValiadtion: # this is components name 
    def __init__(self, config: DataValidationConfig): # it will take my DataValidationConfig
        self.config = config

# now iam below iam going to create validate_all_columns,this is a simple python program
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir) # here iam reading the dataset
            all_cols = list(data.columns) # here iam list down all the columns,it will check or matches the schema.yaml file columns whether all the columns present in schema.yaml file columns  present or not 

            all_schema = self.config.all_schema.keys()

            # if those schema.yaml file columns are present it will return status as true, else it will returnt he status as false, then it will return one txt file inside the artifacts folder which theat txt file contains status decision true or false
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e