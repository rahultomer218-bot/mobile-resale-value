import os
import sys

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"Data transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model trainer")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"Model trainer completed: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                                model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation")
            model_evaluation = ModelEvaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=self.model_evaluation_config
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logger.info(f"Model evaluation completed: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info(f"Model pusher completed: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)

    def run_pipeline(self):
        try:
            logger.info("========== Training Pipeline Started ==========")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact, model_trainer_artifact)
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            logger.info("========== Training Pipeline Completed ==========")
            return model_pusher_artifact
        except Exception as e:
            raise MobileResaleException(e, sys)
