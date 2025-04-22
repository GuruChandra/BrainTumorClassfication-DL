from cnnClassifier.logging import logger
from cnnClassifier.pipeline.stage01_dataingestion_pipeline import DataIngestionPipeline
from cnnClassifier.pipeline.stage02_prepare_baseModel_pipeline import PrepareModelPipeline

logger.info("Welcome to our custom logging...")


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<< \n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e

logger.info(f"X------------------------------------------------------------------X")



STAGE_NAME = "Prepare Model Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    prepare_model = PrepareModelPipeline()
    prepare_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<< \n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e

logger.info(f"X------------------------------------------------------------------X")
