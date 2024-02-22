import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"  # here level name means whether the information related log or its bug level it will save them , and module means which module we are running the code it will save them, and message means simple message string 

log_dir = "logs" # here iam creating one log folder and inside that iam creating thr running log
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig( # nad here iam calling the basics config method
    level= logging.INFO, # and here iam initilaziting everything
    format= logging_str,

    handlers=[  # and calling these below 2 methods whicha are file handler and stream handler 
        logging.FileHandler(log_filepath), # file handler it will create this log folder inside that it will save all the logging 
        logging.StreamHandler(sys.stdout)  # and the stream handler it will print my log in my terminal, as we can see while we priniting the template.py it was executing thr log alsoin the terminal
    ]
)

logger = logging.getLogger("mlProjectLogger")