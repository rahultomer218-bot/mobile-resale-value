import os
import sys
import numpy as np
import dill
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MobileResaleException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models) -> dict:
        try:
            report = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                report[name] = score
                logger.info(f"{name} R2 Score: {score}")
            return report
        except Exception as e:
            raise MobileResaleException(e, sys)

    def get_best_model(self, report: dict):
        try:
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            return best_model_name, best_model_score
        except Exception as e:
            raise MobileResaleException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")

            # Load transformed data
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path, allow_pickle=True)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path, allow_pickle=True)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor()
            }

            # Evaluate all models
            report = self.evaluate_models(X_train, y_train, X_test, y_test, models)
            logger.info(f"Model evaluation report: {report}")

            # Get best model
            best_model_name, best_model_score = self.get_best_model(report)
            logger.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Check if best model score is acceptable
            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"No model found with expected accuracy. Best score: {best_model_score}")

            # Train best model on full data
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metric = ClassificationMetricArtifact(
                f1_score=r2_score(y_train, y_train_pred),
                precision_score=mean_absolute_error(y_train, y_train_pred),
                recall_score=mean_squared_error(y_train, y_train_pred)
            )

            test_metric = ClassificationMetricArtifact(
                f1_score=r2_score(y_test, y_test_pred),
                precision_score=mean_absolute_error(y_test, y_test_pred),
                recall_score=mean_squared_error(y_test, y_test_pred)
            )

            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                dill.dump(best_model, f)

            logger.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            raise MobileResaleException(e, sys)
