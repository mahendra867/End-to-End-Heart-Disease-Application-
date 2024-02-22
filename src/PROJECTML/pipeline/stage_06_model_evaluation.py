from PROJECTML.config.configuration import ConfigurationManager
from PROJECTML.components.model_evaluation import ModelEvaluation
from PROJECTML import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager() # here i have initlized my ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config() # her eiam getiing my get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config) # here iam passing this  model_evaluation_config to my ModelEvalaution
        model_evaluation_config.model_evaluation()
        #model_evaluation_config.log_into_mlflow() # here iam starting this log_into_mlflow()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e