"""
This module implements a FastAPI application for performing model inference 
using a pre-trained machine learning model. It defines endpoints for root access 
and making predictions based on input data. The input data is expected to adhere 
to a specified schema defined by the InputData Pydantic model. Logging messages 
are saved to a log file named 'main_log.log' in the same directory as the script.

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data, load_data
from starter.ml.model import inference


# Get the absolute path of the directory containing the current script
def find_root_directory():
    """
    Find main dir
    """
    current_dir = os.getcwd()

    # Search for main.py recursively from the current directory
    while current_dir != '/':
        main_file = os.path.join(current_dir, 'main.py')
        if os.path.isfile(main_file):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If main.py is not found, return None
    return None

root_dir = find_root_directory()
data_dir = os.path.join(root_dir, "data")
data_fpath = os.path.join(data_dir, "census_v2.csv")
data, str_columns = load_data(data_fpath)

# Get cat features
cat_features = [x for x in str_columns if x != "salary"]

# Define input data model
class InputData(BaseModel):
    """
    Pydantic model representing input data for model inference.
    
    Attributes:
        age (int): Age of the individual.
        workclass (str): Workclass of the individual.
        fnlgt (int): Final weight estimation for the individual.
        education (str): Education level of the individual.
        education_num (int): Numeric representation of education level.
        marital_status (str): Marital status of the individual.
        occupation (str): Occupation of the individual.
        relationship (str): Relationship status of the individual.
        race (str): Race of the individual.
        sex (str): Gender of the individual.
        capital_gain (int): Capital gains of the individual.
        capital_loss (int): Capital losses of the individual.
        hours_per_week (int): Hours worked per week by the individual.
        native_country (str): Native country of the individual.
    
    Config:
        schema_extra (dict): Additional schema information including an example.
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        """
        Configuration settings for the InputData Pydantic model.
        
        This class defines additional schema extra information for the InputData model,
        including an example input data dictionary.
        
        Attributes:
            schema_extra (dict): Additional schema information including an example.
        """
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Divorced",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

# Set up logging
logging.basicConfig(filename="main_log.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create FastAPI app
app = FastAPI()

# Load models
model_dir = os.path.join(root_dir, "model")
model = joblib.load(os.path.join(model_dir, "model.pkl"))
encoder = joblib.load(os.path.join(model_dir, "encoder.pkl"))
lb = joblib.load(os.path.join(model_dir, "lb.pkl"))

# Define root endpoint
@app.get("/")
async def root():
    """Welcome message."""
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to this amazing app!"}

# Define prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData) -> str:
    """
    Perform model inference.
    
    Args:
        input_data (InputData): Input data for prediction.
    
    Returns:
        str: Predicted output label.
    """
    logging.info("Prediction endpoint accessed.")
    logging.info("Input data: %s", input_data)

    # Get the input data from the request data
    data_ = {
        "age": input_data.age,
        "workclass": input_data.workclass,
        "fnlwgt": input_data.fnlgt,
        "education": input_data.education,
        "education-num": input_data.education_num,
        "marital-status": input_data.marital_status,
        "occupation": input_data.occupation,
        "relationship": input_data.relationship,
        "race": input_data.race,
        "sex": input_data.sex,
        "capital-gain": input_data.capital_gain,
        "capital-loss": input_data.capital_loss,
        "hours-per-week": input_data.hours_per_week,
        "native-country": input_data.native_country
    }
    input_df = pd.DataFrame([data_])

    # Process the data
    x, _, _, _ = process_data(
        input_df, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )

    # Inference
    y = inference(model=model, x=x)
    output = lb.inverse_transform(y)[0].strip()

    logging.info("Predicted output: %s", output)
    return output
