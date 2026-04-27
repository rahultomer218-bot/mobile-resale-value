import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataIngestionArtifact, ClassificationMetricArtifact


class ModelEvaluation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig()):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise MobileResaleException(e, sys)

    def get_best_model(self):
        try:
            model_path = self.model_trainer_artifact.trained_model_file_path
            if not os.path.exists(model_path):
                return None
            with open(model_path, "rb") as f:
                model = dill.load(f)
            return model
        except Exception as e:
            raise MobileResaleException(e, sys)

    def evaluate_model(self, model, X_test, y_test) -> ClassificationMetricArtifact:
        try:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            return ClassificationMetricArtifact(
                f1_score=r2,
                precision_score=mae,
                recall_score=mse
            )
        except Exception as e:
            raise MobileResaleException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation")

            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            TARGET_COLUMN = "Predicted_Current_Value_INR"

            X_test = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test = test_df[TARGET_COLUMN]

            # Load preprocessor
            preprocessor_path = "artifacts/data_transformation/transformed_object/preprocessor.pkl"
            with open(preprocessor_path, "rb") as f:
                preprocessor = dill.load(f)

            X_test_transformed = preprocessor.transform(X_test)

            # Load trained model
            trained_model = self.get_best_model()

            # Evaluate trained model
            trained_metric = self.evaluate_model(trained_model, X_test_transformed, y_test)
            logger.info(f"Trained model R2 Score: {trained_metric.f1_score}")
            logger.info(f"Trained model MAE: {trained_metric.precision_score}")
            logger.info(f"Trained model MSE: {trained_metric.recall_score}")

            # Check if model is accepted
            is_model_accepted = trained_metric.f1_score >= self.model_evaluation_config.changed_threshold_score

            # Save evaluation report
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            report_content = f"""
Model Evaluation Report
=======================
R2 Score  : {trained_metric.f1_score}
MAE       : {trained_metric.precision_score}
MSE       : {trained_metric.recall_score}
Accepted  : {is_model_accepted}
"""
            with open(self.model_evaluation_config.report_file_path, "w") as f:
                f.write(report_content)

            logger.info(f"Model evaluation report saved at: {self.model_evaluation_config.report_file_path}")

            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=trained_metric.f1_score,
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                best_model_metric_artifact=trained_metric
            )

        except Exception as e:
            raise MobileResaleException(e, sys)
