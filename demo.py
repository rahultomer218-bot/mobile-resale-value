from dotenv import load_dotenv
load_dotenv()

from src.pipeline.prediction_pipeline import MobileData, PredictionPipeline

# Sample input
mobile = MobileData(
    brand="OnePlus",
    model_name="OnePlus 10R",
    year_of_launch=2022,
    purchase_price_inr=88004
)

# Get dataframe
df = mobile.get_data_as_dataframe()
print("Input Data:")
print(df)

# Predict
pipeline = PredictionPipeline()
predicted_price = pipeline.predict(df)
print(f"\nPredicted Resale Price: ₹{predicted_price:,.2f}")
