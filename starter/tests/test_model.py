"""
Module for testing data processing and model inference functions.

This module contains pytest functions to test the data processing
and model inference functionality provided by the `starter.ml` package.

Author: Fu Cheng
Date: Jan 2024
"""
import os
import sys
sys.path.append('../')
import pytest
import joblib
from sklearn.model_selection import train_test_split
from starter.ml.data import load_data, process_data
from starter.ml.model import inference


@pytest.fixture(scope="module")
def loaded_data():
    """Load data and string column names."""
    # Code to load in the data.
    csv_data_path = "./data/census_v2.csv"
    return load_data(csv_data_path)


def test_load_data(loaded_data):
    """Test loading data and string column names."""
    # Load data and string column names
    data, str_columns = loaded_data

    # Test data shape
    assert data.shape[0] > 0
    assert data.shape[1] > 0

    # Test string columns
    assert isinstance(str_columns, list)
    assert all(isinstance(col, str) for col in str_columns)


def test_process_data(loaded_data):
    """
    Test data train test split
    """
    # Load data
    data, str_columns = loaded_data

    # Split into train and test datasets
    train, test = train_test_split(data, test_size=0.2, random_state=0)

    # Process data
    cat_features = [x for x in str_columns if x != "salary"]
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    x_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Test dataset shape
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(data)


def test_inference(loaded_data):
    """
    Test inference function
    """
    # Load data and model
    data, str_columns = loaded_data
    model = joblib.load(r"./model/model.pkl")
    train, test = train_test_split(data, test_size=0.2, random_state=0)

    # Process data
    cat_features = [x for x in str_columns if x != "salary"]
    _, _, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    x_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    y_preds = inference(model, x_test)
    assert len(y_preds) == len(test)
