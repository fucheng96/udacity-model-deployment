"""
Functions used in model training in the machine learning pipeline

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier


# Optional: implement hyperparameter tuning.
def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Instantiate model
    gbc = GradientBoostingClassifier(random_state=0)

    # Perform model fitting
    gbc.fit(x_train, y_train)

    return gbc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, x):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    x : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(x)
