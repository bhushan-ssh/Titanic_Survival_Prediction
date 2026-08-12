# Titanic Survival Prediction



---

## Overview

**Titanic Survival Prediction** is a machine learning project that predicts whether a passenger survived the Titanic disaster based on demographic, passenger, and travel-related information.

The project follows a complete machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, model development, model comparison, cross-validation, hyperparameter tuning, model interpretation, and Kaggle submission generation.

Four classification algorithms were evaluated:

* Logistic Regression
* SGD Classifier
* Decision Tree
* Random Forest

Among the evaluated models, **Logistic Regression achieved the highest validation accuracy of 81.56%** and was selected as the final model.

---

## Objective

The primary objective is to build a machine learning model capable of predicting passenger survival using the information available in the Titanic dataset.

The project focuses on:

* Understanding the dataset and its underlying patterns
* Performing exploratory data analysis
* Identifying and handling missing values
* Creating meaningful engineered features
* Preparing categorical and numerical variables
* Comparing multiple classification algorithms
* Evaluating model performance using multiple metrics
* Performing cross-validation
* Optimizing the selected model using hyperparameter tuning
* Interpreting model coefficients
* Training the final model using the complete training dataset
* Generating predictions for the Kaggle test dataset
* Creating a Kaggle-compatible submission file

---

## Dataset

The project uses the standard Titanic dataset provided by the Kaggle Titanic competition.

The training dataset contains **891 passenger records** and **12 columns**.

The test dataset contains **418 passenger records** and **11 columns**.

The training dataset contains the target variable `Survived`, while the test dataset is used for generating final predictions.

### Dataset Features

| Feature       | Description                                               |
| ------------- | --------------------------------------------------------- |
| `PassengerId` | Unique passenger identifier                               |
| `Survived`    | Target variable indicating whether the passenger survived |
| `Pclass`      | Passenger class                                           |
| `Name`        | Passenger name                                            |
| `Sex`         | Passenger gender                                          |
| `Age`         | Passenger age                                             |
| `SibSp`       | Number of siblings or spouses aboard                      |
| `Parch`       | Number of parents or children aboard                      |
| `Ticket`      | Passenger ticket number                                   |
| `Fare`        | Passenger fare                                            |
| `Cabin`       | Cabin information                                         |
| `Embarked`    | Port where the passenger embarked                         |

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Did not survive |
| `1`   | Survived        |

---

## Project Architecture

The project follows a structured end-to-end machine learning pipeline.

```text
                         Titanic Dataset
                                |
                                v
                     Exploratory Data Analysis
                                |
                                v
                       Data Preprocessing
                                |
                +---------------+---------------+
                |                               |
                v                               v
        Missing Value Handling          Feature Engineering
                |                               |
                +---------------+---------------+
                                |
                                v
                     Feature Transformation
                                |
                                v
                    Train / Validation Split
                                |
                                v
                       Feature Scaling
                                |
                                v
                       Model Training
                                |
              +-----------------+------------------+
              |                 |                  |
              v                 v                  v
       Logistic Regression   Decision Tree    Random Forest
              |
              +--------------------+
                                   |
                                   v
                           Model Comparison
                                   |
                                   v
                         Cross-Validation
                                   |
                                   v
                        Hyperparameter Tuning
                                   |
                                   v
                            Final Model
                                   |
                                   v
                     Complete Training Dataset
                                   |
                                   v
                         Test Data Prediction
                                   |
                                   v
                          submission.csv
```

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure, distribution, and relationships within the Titanic dataset.

The analysis includes:

* Dataset dimensions
* Data types
* Statistical summaries
* Missing-value analysis
* Survival distribution
* Survival by gender
* Survival by passenger class
* Age distribution
* Fare distribution
* Age versus survival
* Fare versus passenger class
* Numerical feature correlations

Visualizations were used to identify relationships between passenger characteristics and survival.

---

## Survival Analysis

One of the strongest patterns identified during exploratory analysis was the relationship between gender and survival.

| Gender | Survival Rate |
| ------ | ------------: |
| Female |        74.20% |
| Male   |        18.89% |

The substantial difference indicates that `Sex` is one of the most informative features for predicting survival.

Passenger class also showed an important relationship with survival, providing additional predictive information for the models.

---

## Missing Value Analysis

The original dataset contains missing values in several important columns.

| Feature        | Missing Values |
| -------------- | -------------: |
| `Age`          |            177 |
| `Cabin`        |            687 |
| `Embarked`     |              2 |
| `Fare`         |              0 |
| Other features |              0 |

The project handles these missing values rather than removing large portions of the dataset.

### Missing Value Strategy

| Feature    | Approach          |
| ---------- | ----------------- |
| `Age`      | Median imputation |
| `Embarked` | Mode imputation   |
| `Fare`     | Median imputation |
| `Cabin`    | Unknown category  |

This approach preserves the available passenger information while ensuring that the machine learning models receive complete input data.

---

## Feature Engineering

Feature engineering was used to extract additional information from the original Titanic features.

### FamilySize

A new `FamilySize` feature was created from the number of siblings, spouses, parents, and children traveling with the passenger.

This combines multiple family-related variables into a single representation of the passenger's family group.

### IsAlone

The `IsAlone` feature identifies passengers who were traveling without family members.

This provides the model with a direct representation of whether the passenger was traveling alone or with family.

### Title

Passenger titles were extracted from the `Name` feature.

Examples include:

* Mr
* Mrs
* Miss
* Master
* Dr
* Rev

Rare titles were grouped together, while equivalent titles were normalized.

This feature captures additional information contained within passenger names without requiring the complete name as a model feature.

### CabinKnown

A `CabinKnown` feature was created to indicate whether cabin information was available.

Rather than using the raw cabin number, this feature captures whether a passenger had recorded cabin information.

---

## Feature Transformation

Categorical features were converted into numerical representations before model training.

The following features were encoded:

* `Sex`
* `Embarked`
* `Title`

Numerical features were standardized before being provided to models that benefit from feature scaling.

---

## Feature Selection

After preprocessing and feature engineering, the following features were used for model training:

| Feature      |
| ------------ |
| `Pclass`     |
| `Sex`        |
| `Age`        |
| `SibSp`      |
| `Parch`      |
| `Fare`       |
| `Embarked`   |
| `FamilySize` |
| `IsAlone`    |
| `Title`      |
| `CabinKnown` |

The following original columns were excluded from the final model input:

* `PassengerId`
* `Name`
* `Ticket`
* `Cabin`

`PassengerId` is an identifier rather than a meaningful predictive feature, while the information from `Name` and `Cabin` was represented through engineered features.

---

## Train and Validation Split

The labelled training data was divided into training and validation subsets.

| Dataset    | Samples |
| ---------- | ------: |
| Training   |     712 |
| Validation |     179 |

The split uses:

* 80% training data
* 20% validation data
* Fixed random state for reproducibility
* Stratification based on the target variable

This provides a consistent validation environment for comparing different models.

---

## Machine Learning Models

Four classification algorithms were evaluated.

### Logistic Regression

Logistic Regression was used as a linear classification model and ultimately achieved the best validation performance.

It was also selected for further hyperparameter optimization and interpretability analysis.

### SGD Classifier

A Stochastic Gradient Descent classifier using logistic loss was evaluated as an alternative linear classification approach.

### Decision Tree

A Decision Tree classifier was included to evaluate nonlinear relationships between passenger features and survival.

### Random Forest

Random Forest was evaluated as an ensemble of decision trees.

It provides a nonlinear alternative to the linear Logistic Regression model and can capture more complex feature relationships.

---

## Model Comparison

The models were compared using validation accuracy.

| Rank | Model                   | Validation Accuracy |
| ---: | ----------------------- | ------------------: |
|    1 | **Logistic Regression** |          **81.56%** |
|    2 | Random Forest           |              79.89% |
|    3 | Decision Tree           |              76.54% |
|    4 | SGD Classifier          |              75.42% |

### Best Performing Model

**Logistic Regression**

Validation Accuracy:

**81.56%**

Logistic Regression achieved the highest validation accuracy among the four evaluated models and was therefore selected for further optimization.

---

## Cross-Validation

Five-fold cross-validation was performed on the Logistic Regression model to evaluate the consistency of its performance across different subsets of the training data.

The observed cross-validation scores were approximately:

| Fold | Accuracy |
| ---: | -------: |
|    1 |   79.72% |
|    2 |   75.52% |
|    3 |   81.69% |
|    4 |   81.69% |
|    5 |   79.58% |

The overall result was:

| Metric             |     Result |
| ------------------ | ---------: |
| Mean CV Accuracy   | **79.64%** |
| Standard Deviation |  **2.25%** |

The relatively small variation across folds indicates reasonably consistent model performance.

---

## Model Evaluation

The selected Logistic Regression model was evaluated using multiple metrics.

### Accuracy

The final validation accuracy was:

**81.56%**

This represents the percentage of validation passengers correctly classified by the model.

### ROC AUC

The model achieved:

**ROC AUC: 0.8581**

ROC AUC measures the model's ability to distinguish between surviving and non-surviving passengers across different classification thresholds.

### Confusion Matrix

A confusion matrix was used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

This provides a more detailed view of the classification errors made by the model.

### Classification Report

Precision, recall, and F1-score were also calculated for the two survival classes.

These metrics provide additional information about class-specific performance beyond overall accuracy.

---

## Model Interpretability

One of the advantages of Logistic Regression is its interpretability.

The model coefficients were analyzed to understand how the engineered features contribute to the prediction.

The learned coefficients included:

| Feature      | Coefficient |
| ------------ | ----------: |
| `Sex`        |   -1.251817 |
| `Pclass`     |   -0.630029 |
| `Age`        |   -0.433228 |
| `IsAlone`    |   -0.296661 |
| `SibSp`      |   -0.282191 |
| `FamilySize` |   -0.227047 |
| `Embarked`   |   -0.152093 |
| `Title`      |   -0.098280 |
| `Parch`      |   -0.073333 |
| `Fare`       |    0.039990 |
| `CabinKnown` |    0.356719 |

The coefficients were also visualized to provide an intuitive representation of feature influence.

This interpretability is an important advantage of the selected model compared with more complex black-box approaches.

---

## Hyperparameter Tuning

After model comparison, Logistic Regression was optimized using Grid Search with five-fold cross-validation.

The search focused on:

* Regularization strength
* Optimization solver

The best configuration identified during the search was:

| Parameter | Selected Value |
| --------- | -------------- |
| `C`       | `1`            |
| `solver`  | `lbfgs`        |

The optimized Logistic Regression model was then selected as the final model.

---

## Final Model Pipeline

The final model follows the complete pipeline below:

```text
Raw Passenger Data
        |
        v
Missing Value Handling
        |
        v
Feature Engineering
        |
        +------------------+
        |                  |
        v                  v
   Family Features     Passenger Title
        |                  |
        +--------+---------+
                 |
                 v
         Feature Selection
                 |
                 v
        Categorical Encoding
                 |
                 v
          Feature Scaling
                 |
                 v
       Optimized Logistic Regression
                 |
                 v
        Survival Prediction
```

---

## Final Training

After hyperparameter tuning, the final Logistic Regression model was retrained using the complete labelled training dataset.

The complete training data was used so that the final model could learn from all available labelled passenger records before generating predictions for the Kaggle test dataset.

The same preprocessing procedure was applied to the test data to ensure consistency between training and prediction.

---

## Kaggle Submission

The final model generates survival predictions for the Kaggle test dataset.

The submission contains:

| Column        | Description              |
| ------------- | ------------------------ |
| `PassengerId` | Passenger identifier     |
| `Survived`    | Predicted survival class |

The final output file is:

**`submission.csv`**

This file follows the required structure for the Kaggle Titanic competition.

---

## End-to-End Workflow

```text
                    DATA
                     |
                     v
          Exploratory Data Analysis
                     |
                     v
             Data Preprocessing
                     |
                     v
            Feature Engineering
                     |
                     v
             Feature Selection
                     |
                     v
            Train / Validation
                     |
                     v
             Feature Scaling
                     |
                     v
        +------------+------------+
        |            |            |
        v            v            v
   Logistic       Decision      Random
  Regression       Tree         Forest
        |            |            |
        +------------+------------+
                     |
                     v
             Model Comparison
                     |
                     v
             Best Model Selected
                     |
                     v
           Cross-Validation
                     |
                     v
          Hyperparameter Tuning
                     |
                     v
              Final Model
                     |
                     v
        Complete Training Dataset
                     |
                     v
             Test Prediction
                     |
                     v
             submission.csv
```

---

## Project Structure

```text
Titanic_Survival_Prediction/
|
├── titanic-survival-prediction.ipynb
├── submission.csv
├── README.md
└── requirements.txt
```

### Notebook

Contains the complete workflow from data exploration to final Kaggle predictions.

### submission.csv

Contains the final predictions generated for the Kaggle test dataset.

### README.md

Project documentation, methodology, results, and findings.

### requirements.txt

Contains the Python dependencies required to reproduce the project.

---

## Technology Stack

| Category             | Technology                                             |
| -------------------- | ------------------------------------------------------ |
| Programming Language | Python                                                 |
| Data Processing      | Pandas, NumPy                                          |
| Visualization        | Matplotlib, Seaborn                                    |
| Machine Learning     | Scikit-learn                                           |
| Classification       | Logistic Regression, SGD, Decision Tree, Random Forest |
| Model Selection      | Cross-Validation, Grid Search                          |
| Competition Platform | Kaggle                                                 |

---

## Key Findings

### Gender

Gender was one of the strongest predictors of survival.

Female passengers had a survival rate of approximately **74.20%**, compared with approximately **18.89%** for male passengers.

### Passenger Class

Passenger class showed a strong relationship with survival, making `Pclass` an important feature in the final model.

### Family Structure

The engineered `FamilySize` and `IsAlone` features allowed the model to capture information about passengers traveling with or without family members.

### Cabin Information

Although the raw cabin information contains a large number of missing values, the availability of cabin information itself was converted into the `CabinKnown` feature.

### Model Selection

Logistic Regression produced the strongest validation accuracy among the evaluated models.

Its combination of performance, simplicity, and interpretability made it the final selected model.

---

## Results Summary

| Metric                              |                  Result |
| ----------------------------------- | ----------------------: |
| Best Model                          | **Logistic Regression** |
| Validation Accuracy                 |              **81.56%** |
| Mean Cross-Validation Accuracy      |              **79.64%** |
| Cross-Validation Standard Deviation |               **2.25%** |
| ROC AUC                             |              **0.8581** |
| Best `C`                            |                   **1** |
| Best Solver                         |               **lbfgs** |

---

## Limitations

The Titanic dataset is relatively small, so validation results can vary depending on the data split.

The project primarily evaluates classical machine learning algorithms. More advanced ensemble and gradient boosting methods could potentially improve predictive performance.

The current feature engineering strategy also provides opportunities for further experimentation.

---

## Future Improvements

Potential improvements include:

* More extensive feature engineering
* Ticket group analysis
* Fare-per-person features
* Cabin deck extraction
* Age-group features
* Family survival features
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Support Vector Machines
* More extensive hyperparameter optimization
* Ensemble learning
* Automated preprocessing pipelines
* Cross-validation-based model comparison
* Detailed error analysis

---

## Reproducibility

The project uses a fixed random state for the train-validation split and model initialization where applicable.

This helps ensure that the primary experiments can be reproduced under the same environment and dataset configuration.

To reproduce the project:

1. Clone the repository.
2. Install the required Python dependencies.
3. Obtain the Titanic dataset.
4. Place the dataset in the expected Kaggle or local data location.
5. Run the notebook from beginning to end.
6. Review the model comparison and evaluation results.
7. Generate the final `submission.csv`.

---

## Conclusion

Titanic Survival Prediction demonstrates a complete machine learning workflow applied to a binary classification problem.

The project progresses from exploratory data analysis and data preprocessing to feature engineering, model comparison, cross-validation, hyperparameter optimization, model interpretation, and final Kaggle prediction.

Four classification algorithms were evaluated, with Logistic Regression achieving the strongest validation performance.

The final model achieved:

* **81.56% validation accuracy**
* **79.64% mean cross-validation accuracy**
* **0.8581 ROC AUC**

The project demonstrates the importance of data preprocessing, meaningful feature engineering, model comparison, and systematic evaluation when developing a machine learning solution.

---

## Author

**Bhushan Dattatray Sonawane**

Roll No.: 23f2003210

**Project:** Titanic Survival Prediction

**Focus:** Machine Learning and Data Analysis
