# Titanic Survival Prediction

Predict passenger survival on the Titanic using Machine Learning.

## Project Overview

This project is based on the Kaggle **Titanic - Machine Learning from Disaster** competition. The objective is to build a classification model that predicts whether a passenger survived using demographic and travel information.

## Dataset

The dataset includes:

- train.csv
- test.csv
- gender_submission.csv

Features include:

- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

Target:

- Survived (0 = No, 1 = Yes)

## Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Prediction Generation
8. Kaggle Submission

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost (Optional)

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Results

The trained model predicts survival for the Kaggle test dataset and generates a submission file.

## Project Structure

```
Titanic_Survival_Prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── submissions/
├── README.md
├── requirements.txt
└── LICENSE
```

## Competition

Kaggle: Titanic - Machine Learning from Disaster

## Author

Bhushan Sonawane
