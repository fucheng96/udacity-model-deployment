"""
Test module for FastAPI application endpoints.

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
import os
import sys
sys.path.append(os.path.abspath('..'))
from main import app
from fastapi.testclient import TestClient


# Instantiate the testing client with our app.
client = TestClient(app)

def test_get_root():
    """
    Test GET root endpoint.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[0] == "Hello world!"


def test_post_inference():
    """
    Test model inference with valid query.
    """
    sample = {
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
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post_inference_false_query():
    """
    Test model inference with invalid query.
    """
    sample = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male"
    }
    r = client.post("/predict", json=sample)
    assert 'capital_gain' not in r.json()
    assert 'capital_loss' not in r.json()
    assert 'hours_per_week' not in r.json()
    assert 'native_country' not in r.json()
