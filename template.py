import os
from pathlib import Path
import logging # i want to see my log in the terminal thats why this log is need 
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:') # here iam initilizing one log string which it display the logg in the terminal like information logg level in format as per the time we exectue the code  w.r.t its message if code contain errors it returns the message error 

project_name='PROJECTML' # this is the project name which i have given as 'mlproject_with_mlflow', usually we create one folder called src inside that src folder it will create 'mlproject_with_mlflow' inside this folder we all create all of the components 


# here i have taken one list which it contain the list of files or records 
list_of_files = [                                 
    ".github/workflows/.gitkeep",  # here in the 1st record iam creating 1 folder named as .github then workflows then .gitkeep 
    f"src/{project_name}/__init__.py",  # this is how we are arrainging the many folders config,components, utils etc which consist of constructuer file like __init__.py  inside the projectname folder which it is present inside the src folder 
    f"src/{project_name}/components/__init__.py", # here __init__.py is the constructor file which it makes the components folder as a local package because i want to import something from the component folder thats why we need to ready with constructure file which is __init__.py
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"


]

# here by using the below code iam trying to convert the above list of records or folders into paths like we are actually separating our folder into filedir = f"src/{project_name}  and filepath=/components/__init__.py
for filepath in list_of_files:  # here i have created the for file and passing the listof files folder which i have created above 
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)  # filedir = f"src/{project_name}, filepath=/components/__init__.py


    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): # (not os.path.exists(filepath)) checks if the file specified by filepath does not exist. (os.path.getsize(filepath) == 0) checks if the size of the file is zero bytes., If this condition is True, the code enters the block following the if statement:
        with open(filepath,"w") as f:   # Here, f is used as a variable name. It's a common convention to use f as a variable name for file handles. The open function is used to open a file, and it returns a file object, which is assigned to the variable f in this case. The "w" argument specifies that the file should be opened in write mode. The with statement is used to ensure that the file is properly closed after writing is done
            pass  # This block of code opens the file specified by filepath in write mode ("w"). If the file doesn't exist, it will be created. The pass statement does nothing, and then a log message is generated indicating that an empty file is being created.
            logging.info(f"creating empty file: {filepath}")

    else: # If the condition is False, meaning that the file exists and has a non-zero size, the code enters the else block:
        logging.info(f"{filename} is already exists")