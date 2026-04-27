import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "mobile_resale_value"

list_of_files = [
    # GitHub Actions
    ".github/workflows/.gitkeep",

    # Config
    "config/config.yaml",

    # Mobile Resale Value (core package)
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",

    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/mongo_db_connection.py",

    f"{project_name}/constants/__init__.py",

    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",

    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",

    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logging.py",

    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",

    # Notebooks
    "notebook/mobile_data.csv",
    "notebook/mongoDB_demo.ipynb",
    "notebook/research.ipynb",

    # Source (API / UI)
    "src/__init__.py",
    "src/components/__init__.py",
    "src/configuration/__init__.py",
    "src/constants/__init__.py",
    "src/entity/__init__.py",
    "src/exception/__init__.py",
    "src/logger/__init__.py",
    "src/pipeline/__init__.py",
    "src/utils/__init__.py",

    # Artifacts (generated during training pipeline)
    "artifacts/data_ingestion/raw_data/.gitkeep",
    "artifacts/data_ingestion/ingested/train.csv",
    "artifacts/data_ingestion/ingested/test.csv",

    "artifacts/data_transformation/transformed/train.npy",
    "artifacts/data_transformation/transformed/test.npy",
    "artifacts/data_transformation/transformed_object/preprocessor.pkl",

    "artifacts/model_trainer/trained_model/model.pkl",

    "artifacts/model_evaluation/.gitkeep",

    "artifacts/model_pusher/.gitkeep",

    # Root-level files
    "app.py",
    "main.py",
    "demo.py",
    "Dockerfile",
    ".dockerignore",
    ".env",
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "schema.yaml",
    "README.md",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    # Create directories if they don't exist
    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    # Create file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass  # Create empty file
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")