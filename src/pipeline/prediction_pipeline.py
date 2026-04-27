import os
import sys
import pandas as pd
import dill

from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException


class MobileData:
    def __init__(self,
                 brand: str,
                 model_name: str,
                 year_of_launch: int,
                 purchase_price_inr: float):
        try:
            self.brand = brand
            self.model_name = model_name
            self.year_of_launch = year_of_launch
            self.purchase_price_inr = purchase_price_inr
        except Exception as e:
            raise MobileResaleException(e, sys)

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Brand": [self.brand],
                "Model_Name": [self.model_name],
                "Year_of_Launch": [self.year_of_launch],
                "Purchase_Price_INR": [self.purchase_price_inr]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise MobileResaleException(e, sys)


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("saved_models", "model.pkl")
        self.preprocessor_path = os.path.join(
            "artifacts", "data_transformation", "transformed_object", "preprocessor.pkl"
        )

    def load_model(self, path: str):
        try:
            with open(path, "rb") as f:
                return dill.load(f)
        except Exception as e:
            raise MobileResaleException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> float:
        try:
            logger.info("Loading model and preprocessor")
            model = self.load_model(self.model_path)
            preprocessor = self.load_model(self.preprocessor_path)

            logger.info("Transforming input data")
            transformed_data = preprocessor.transform(dataframe)

            logger.info("Making prediction")
            prediction = model.predict(transformed_data)

            return prediction[0]

        except Exception as e:
            raise MobileResaleException(e, sys)
