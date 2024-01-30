# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    - Model Developer: Fu Cheng
    - Model Date: 29 Jan 2024
    - Model Version: 1.0.0
    - Model Type: Random Forest Classifier
    - Model Library: scikit-learn
    - Model License: MIT License 

## Intended Use
    - Primary Intended Uses (Primary Purpose): This model aims to predict if someone earns more than $50,000 a year using published census data.
    - Primary Intended Users (Who It's For): It's designed for managers, HR professionals, and anyone curious about job market salary trends.
    - Out-Of-Scope Use Cases (What It Can't Do): The model might not be very accurate in certain job markets where salaries vary a lot or where there isn't much data.

## Training Data
    The training data used for this model includes various socio-demographic attributes from the [1994 Census database](https://archive.ics.uci.edu/ml/datasets/census+income). These attributes encompass features such as age, education level, marital status, occupation, race, and others, as described below:

    1. age: Continuous variable representing the age of individuals.
    2. workclass: Categorical variable indicating the type of employment, including options such as Private, Self-emp-not-inc, and others.
    3. fnlwgt: Continuous variable representing final weight, an estimation of the number of people the census believes the entry represents.
    4. education: Categorical variable indicating the highest level of education completed.
    5. education-num: Continuous variable representing the numerical education level.
    6. marital-status: Categorical variable indicating the marital status of individuals.
    7. occupation: Categorical variable indicating the type of occupation.
    8. relationship: Categorical variable indicating the relationship status of individuals.
    9. race: Categorical variable indicating the race of individuals.
    10. sex: Categorical variable indicating the gender of individuals.
    11. capital-gain: Continuous variable representing capital gains.
    12. capital-loss: Continuous variable representing capital losses.
    13. hours-per-week: Continuous variable representing the number of hours worked per week.
    14. native-country: Categorical variable indicating the native country of individuals.

## Evaluation Data
    - The evaluation data was derived from splitting the original dataset into training and test sets using an 80:20 ratio.
    - The training set, comprising 80% of the data, was used to train the model, while the test set, comprising 20% of the data, was used to evaluate its performance.
    - This standard practice helps assess the model's generalization capabilities and ensures that it can make accurate predictions on unseen data.

## Metrics
    1. [Precision](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/precision): 0.74
    2. [Recall](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/recall): 0.63
    3. [FBeta](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/f-beta-score): 0.68

## Ethical Considerations
    - The dataset used for training this model was compiled by Barry Becker from the 1994 Census database.
    - Given its age, there may be limitations in its representativeness of contemporary demographics and socio-economic conditions, potentially impacting the model's performance and relevance in present contexts.
    - Careful consideration should be given to these temporal discrepancies when interpreting the model's predictions and ensuring fairness in its deployment.

## Caveats and Recommendations
    - This dataset is sourced from the UCI Machine Learning Repository.
    - While utilizing established datasets can mitigate some ethical concerns, it's essential to remain vigilant about potential biases and inaccuracies inherent in the data collection process.
    - Users should critically assess the dataset's suitability for their specific use case and consider augmenting it with more recent or domain-specific data to enhance the model's robustness and generalizability.