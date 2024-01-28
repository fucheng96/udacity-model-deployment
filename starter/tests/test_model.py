"""
Module for testing data processing and model inference functions.

This module contains pytest functions to test the data processing
and model inference functionality provided by the `starter.ml` package.

Author: Fu Cheng
Date: Jan 2024
"""
import sys
import pytest
import joblib
from sklearn.model_selection import train_test_split
from starter.ml.data_process import load_data, process_data
from starter.ml.model import inference


@pytest.fixture(scope="module")
def get_data_and_str_cols():
    # Code to load in the data.
    sys.path.append('../')
    csv_data_path = "./data/census_v2.csv"
    return load_data(csv_data_path)


@pytest.fixture(scope="module")
def loaded_data():
    data_fpath = "path_to_your_data_file.csv"  # Provide the path to your data file
    data, str_columns = load_data(data_fpath)
    return data, str_columns

def test_load_data_shape(loaded_data):
    data, str_columns = loaded_data
    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_load_data_str_columns(loaded_data):
    data, str_columns = loaded_data
    assert isinstance(str_columns, list)
    assert all(isinstance(col, str) for col in str_columns)


def test_process_data(data):
    """
    Test data train test split
    """
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    # Process data
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    x_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(data)


def test_inference(data):
    """
    Test inference function
    """
    model = joblib.load(r'./model/model.pkl')
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    # Process data
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    x_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    y_preds = inference(model, x_test)
    assert len(y_preds) == len(test)
