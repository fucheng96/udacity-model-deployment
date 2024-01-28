# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    Model Developer: Fu Cheng
    Model Date: 29 Jan 2024
    Model Version: 1.0.0
    Model Type: Gradient Boosting Classifier
    Model Library: scikit-learn
    Model License: MIT License 

## Intended Use
    Primary Intended Uses (Primary Purpose): This model aims to predict if someone earns more than $50,000 a year using published census data.
    Primary Intended Users (Who It's For): It's designed for managers, HR professionals, and anyone curious about job market salary trends.
    Out-Of-Scope Use Cases (What It Can't Do): The model might not be very accurate in certain job markets where salaries vary a lot or where there isn't much data.

## Training Data
    <!--Model input dataset is the Census Income dataset from https://archive.ics.uci.edu/ml/datasets/census+income--!>

## Evaluation Data

## Metrics
    1. [Precision](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/precision): 0.78
    2. [Recall](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/recall): 0.62
    3. [FBeta](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/f-beta-score): 0.69

## Ethical Considerations

## Caveats and Recommendations
