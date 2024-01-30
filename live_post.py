"""
POSTs to the API

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
from typing import Dict
import requests

# API endpoint
URL = "https://census-income-predict.onrender.com/predict"

# Sample data to be sent as JSON
sample: Dict[str, str] = {
    "age": 35,                           # Age of the individual
    "workclass": "Private",              # Workclass of the individual
    "fnlgt": 77516,                      # Final weight
    "education": "HS-grad",              # Education level
    "education_num": 9,                  # Numeric education level
    "marital_status": "Divorced",        # Marital status
    "occupation": "Handlers-cleaners",   # Occupation
    "relationship": "Husband",           # Relationship
    "race": "Black",                     # Race
    "sex": "Male",                       # Gender
    "capital_gain": 0,                   # Capital gain
    "capital_loss": 0,                   # Capital loss
    "hours_per_week": 40,                # Hours worked per week
    "native_country": "United-States"    # Native country
}

# Send POST request to the API with sample data
response = requests.post(URL, json=sample)

# Print status code and response body
print(f"Status code: {response.status_code}")
print(f"Body: {response.json()}")
