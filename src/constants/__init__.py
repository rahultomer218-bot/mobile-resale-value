import os
from datetime import datetime

# MongoDB Constants
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME = "mobile_resale_value"
COLLECTION_NAME = "mobile_data"

# Pipeline Constants
PIPELINE_NAME = "mobile_resale_value"
ARTIFACT_DIR = "artifacts"

# Data Ingestion Constants
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2
DATA_INGESTION_RAW_DATA_DIR = "raw_data"
DATA_INGESTION_FILE_NAME = "mobile_data.csv"

# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

# Model Evaluation Constants
MODEL_EVALUATION_DIR_NAME = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MODEL_EVALUATION_REPORT_NAME = "report.yaml"

# Model Pusher Constants
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = "saved_models"

# Timestamp
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")