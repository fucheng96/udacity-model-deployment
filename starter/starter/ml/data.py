"""
Process the data used in the machine learning pipeline

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_data(csv_data_path: str):
    """Load data from a CSV file and extract string columns.

    Reads the data from the provided CSV file path and extracts
    the columns with string dtype.

    Parameters
    ----------
    csv_data_path : str
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the loaded data.
    str_columns : list
        List of column names containing string dtype.
    """
    # Read data
    df = pd.read_csv(csv_data_path)

    # Get string columns
    str_columns = df.select_dtypes(include=["object"]).columns.tolist()

    return df, str_columns


def process_data(
    x, categorical_features: list, label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    x : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    if label is not None:
        y = x[label]
        x = x.drop([label], axis=1)
    else:
        y = np.array([])

    x_categorical = x[categorical_features].values
    x_continuous = x.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    x = np.concatenate([x_continuous, x_categorical], axis=1)
    return x, y, encoder, lb


def slice_performance(
    test_data: pd.DataFrame,
    model: BaseEstimator,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    compute_model_metrics: Callable[[List, List], Tuple[float, float, float]],
    cat_features: List[str]
) -> None:
    """Calculate performance of the model on slices of the data and write to file.

    Args:
        test_data (DataFrame): Test data.
            The dataframe containing the data to be sliced and evaluated.
        model (BaseEstimator): Trained machine learning model.
            The model to be evaluated on the slices of data.
        encoder (OneHotEncoder): Trained OneHotEncoder.
            The encoder used to transform categorical features.
        lb (LabelBinarizer): Trained LabelBinarizer.
            The label binarizer used to transform target labels.
        compute_model_metrics (Callable): Function to calculate precision, recall, and F1.
            This function should accept true labels and predicted labels
            as input and return precision, recall, and F1 scores.
        cat_features (List[str]): List of categorical feature names.
            The list of column names considered categorical features.

    Returns:
        None
    """
    with open('slice_performance.txt', 'w', encoding="utf-8") as f:
        # Loop through all categorical features
        for cat in cat_features:
            # Loop through each unique value in each categorical feature
            for value in test_data[cat].unique():
                # Get slices of the data
                slice_data = test_data[
                    test_data[cat] == value
                ].reset_index(drop=True)

                # Process the slice_data with the process_data function
                x_test, y_test, _, _ = process_data(
                    slice_data, categorical_features=cat_features,
                    label="salary", training=False, encoder=encoder, lb=lb
                )

                # Predict x_test
                y_pred = model.predict(x_test)
                # Evaluate model metrics
                precision, recall, fbeta = compute_model_metrics(
                    y_test, y_pred
                )

                # Write results to file
                row = f"""{cat}:{value}, Precision:{precision},
                        Recall:{recall}, Fbeta:{fbeta}\n""".format(
                    cat=cat, value=value, precision=round(precision, 2),
                    recall=round(recall, 2), fbeta=round(fbeta, 2)
                )
                f.write(row)
