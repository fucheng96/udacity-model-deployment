"""
Script to train machine learning model

Author: Fu Cheng
Date: Jan 2024
"""
# Import libraries
import os
import pickle
from sklearn.model_selection import train_test_split

# Import functions from other modules
from ml.data import load_data, process_data, slice_performance
from ml.model import train_model, compute_model_metrics, inference


# Load data
# Navigate to data folder
starter_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(starter_dir, "data")

# Import data
data_fpath = os.path.join(data_dir, "census_v2.csv")
data, str_columns = load_data(data_fpath)

# Remove salary from str_columns to get cat_features
cat_features = [x for x in str_columns if x != "salary"]

# Perform train-test split
train, test = train_test_split(data, test_size=0.2)

# Process training data ith the process_data function
x_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function
x_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Model fitting
print("Model training")
model = train_model(x_train, y_train)
print("Model Inference")
y_pred = inference(model, x_test)
print("Model Evaluation")
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"""Precision: {round(precision, 2)}
      Recall: {round(recall, 2)}
      FBeta: {round(fbeta, 2)}""")

# Test on slices of the data
print("Caculate performance of the model on slices of the data")
slice_performance(
    test, model, encoder, lb, compute_model_metrics,
    cat_features=cat_features
)

# Save model
with open(os.path.join(starter_dir, "model/model.pkl"), "wb") as model_file:
    pickle.dump(model, model_file)

# Save encoder
with open(os.path.join(starter_dir, "model/encoder.pkl"), "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Save lb
with open(os.path.join(starter_dir, "model/lb.pkl"), "wb") as lb_file:
    pickle.dump(lb, lb_file)
