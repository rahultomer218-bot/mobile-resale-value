import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.configuration.mongo_db_connection import MongoDBClient
from mobile_resale_value.constants import DATABASE_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MobileResaleException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logger.info("Exporting data from MongoDB to DataFrame")
            collection_name = self.data_ingestion_config.collection_name
            mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            collection = mongo_client.database[collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            logger.info(f"Data exported successfully with shape: {df.shape}")
            return df

        except Exception as e:
            raise MobileResaleException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Saving data to feature store")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Data saved to feature store at: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise MobileResaleException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            logger.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logger.info(f"Train set saved at: {self.data_ingestion_config.training_file_path}")
            logger.info(f"Test set saved at: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise MobileResaleException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion pipeline")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logger.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MobileResaleException(e, sys)
