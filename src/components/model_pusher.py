import os
import sys
import shutil
import dill

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig = ModelPusherConfig()):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise MobileResaleException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher")

            # Check if model is accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                raise Exception("Model is not accepted. Skipping model pusher.")

            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # Save to model pusher dir
            os.makedirs(self.model_pusher_config.model_pusher_dir, exist_ok=True)
            model_pusher_file_path = os.path.join(
                self.model_pusher_config.model_pusher_dir, "model.pkl"
            )
            shutil.copy(trained_model_path, model_pusher_file_path)
            logger.info(f"Model copied to model pusher dir: {model_pusher_file_path}")

            # Save to saved_models dir
            os.makedirs(self.model_pusher_config.saved_model_path, exist_ok=True)
            saved_model_file_path = os.path.join(
                self.model_pusher_config.saved_model_path, "model.pkl"
            )
            shutil.copy(trained_model_path, saved_model_file_path)
            logger.info(f"Model saved to saved_models dir: {saved_model_file_path}")

            logger.info("Model pusher completed successfully")

            return ModelPusherArtifact(
                saved_model_path=saved_model_file_path,
                model_file_path=model_pusher_file_path
            )

        except Exception as e:
            raise MobileResaleException(e, sys)
