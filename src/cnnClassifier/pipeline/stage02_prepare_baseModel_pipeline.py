from cnnClassifier.logging import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel 
from cnnClassifier.config.configuration import ConfigurationManager

class PrepareModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
            
        except Exception as e:
            raise e
