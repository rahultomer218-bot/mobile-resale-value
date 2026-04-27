from dataclasses import dataclass
import os
from mobile_resale_value.constants import *

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, DATA_INGESTION_FILE_NAME)
    training_file_path: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_INGESTED_DIR, "train.csv")
    testing_file_path: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_INGESTED_DIR, "test.csv")
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = COLLECTION_NAME

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "train.npy")
    transformed_test_file_path: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "test.npy")
    transformed_object_file_path: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSOR_OBJECT_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    overfitting_underfitting_threshold: float = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD

@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(ARTIFACT_DIR, MODEL_EVALUATION_DIR_NAME)
    report_file_path: str = os.path.join(ARTIFACT_DIR, MODEL_EVALUATION_DIR_NAME, MODEL_EVALUATION_REPORT_NAME)
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(ARTIFACT_DIR, MODEL_PUSHER_DIR_NAME)
    saved_model_path: str = os.path.join(MODEL_PUSHER_SAVED_MODEL_DIR)
