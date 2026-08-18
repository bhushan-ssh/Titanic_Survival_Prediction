import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


@st.cache_resource
def train_model():
    """
    Reproduces the preprocessing and final Logistic Regression
    workflow from Titanic_Survival_Prediction.ipynb.
    """
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # The notebook concatenates train + test before preprocessing.
    full_data = pd.concat([train, test], axis=0, ignore_index=True)

    # Same missing-value handling as the notebook.
    full_data["Age"] = full_data["Age"].fillna(full_data["Age"].median())
    full_data["Embarked"] = full_data["Embarked"].fillna(
        full_data["Embarked"].mode()[0]
    )
    full_data["Fare"] = full_data["Fare"].fillna(full_data["Fare"].median())
    full_data["Cabin"] = full_data["Cabin"].fillna("Unknown")

    # Same feature engineering.
    full_data["FamilySize"] = (
        full_data["SibSp"] + full_data["Parch"] + 1
    )
    full_data["IsAlone"] = (
        full_data["FamilySize"] == 1
    ).astype(int)

    full_data["Title"] = full_data["Name"].str.extract(
        " ([A-Za-z]+)\.", expand=False
    )

    full_data["Title"] = full_data["Title"].replace(
        [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona"
        ],
        "Rare"
    )
    full_data["Title"] = full_data["Title"].replace("Ms", "Miss")
    full_data["Title"] = full_data["Title"].replace("Mlle", "Miss")
    full_data["Title"] = full_data["Title"].replace("Mme", "Mrs")

    # Same LabelEncoder behavior as the notebook.
    full_data["Sex"] = LabelEncoder().fit_transform(full_data["Sex"])
    full_data["Embarked"] = LabelEncoder().fit_transform(full_data["Embarked"])
    full_data["Title"] = LabelEncoder().fit_transform(full_data["Title"])

    full_data["CabinKnown"] = (
        full_data["Cabin"] != "Unknown"
    ).astype(int)

    full_data.drop(
        ["PassengerId", "Name", "Ticket", "Cabin"],
        axis=1,
        inplace=True
    )

    train_processed = full_data.iloc[: len(train)].copy()

    X = train_processed.drop("Survived", axis=1)
    y = train_processed["Survived"]

    # The notebook fits the final scaler on the complete training data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Same GridSearchCV configuration as the notebook.
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_scaled, y)

    # Same final model construction as the notebook.
    final_model = LogisticRegression(
        **grid.best_params_,
        max_iter=2000,
        random_state=42
    )
    final_model.fit(X_scaled, y)

    # Store medians for the UI defaults / fallback.
    metadata = {
        "age_median": float(train["Age"].median()),
        "fare_median": float(train["Fare"].median()),
        "best_params": grid.best_params_,
        "feature_names": list(X.columns)
    }

    return final_model, scaler, metadata


def make_input(
    pclass, sex, age, sibsp, parch, fare,
    embarked, family_size, is_alone,
    title, cabin_known
):
    # Exact label mappings produced by LabelEncoder for the
    # Titanic categories used in the notebook.
    sex_map = {"Female": 0, "Male": 1}
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    title_map = {
        "Master": 0,
        "Miss": 1,
        "Mr": 2,
        "Mrs": 3,
        "Rare": 4
    }

    return pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_map[sex],
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_map[embarked],
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": title_map[title],
        "CabinKnown": int(cabin_known)
    }])


st.title("🚢 Titanic Survival Prediction")
st.write(
    "An end-to-end Machine Learning application based on my "
    "Titanic Survival Prediction project."
)

st.info(
    "Model: Tuned Logistic Regression • "
    "Validation Accuracy: 81.56% • ROC AUC: 0.8585"
)

try:
    model, scaler, metadata = train_model()
except FileNotFoundError:
    st.error(
        "train.csv or test.csv was not found. "
        "Place both files in the root of the GitHub repository."
    )
    st.stop()

st.subheader("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=metadata["age_median"],
        step=1.0
    )
    sibsp = st.number_input(
        "Siblings / Spouses (SibSp)",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )
    parch = st.number_input(
        "Parents / Children (Parch)",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )

with col2:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=metadata["fare_median"],
        step=1.0
    )
    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )
    title = st.selectbox(
        "Title",
        ["Mr", "Miss", "Mrs", "Master", "Rare"]
    )
    cabin_known = st.selectbox(
        "Cabin Information Available?",
        [False, True]
    )

family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

st.caption(
    f"Engineered features → FamilySize: {family_size} | "
    f"IsAlone: {is_alone}"
)

if st.button("🔮 Predict Survival", use_container_width=True):
    input_df = make_input(
        pclass=pclass,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        embarked=embarked,
        family_size=family_size,
        is_alone=is_alone,
        title=title,
        cabin_known=cabin_known
    )

    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0, 1])

    st.divider()

    if prediction == 1:
        st.success("🟢 Prediction: Passenger likely **SURVIVED**")
    else:
        st.error("🔴 Prediction: Passenger likely **DID NOT SURVIVE**")

    st.metric(
        "Estimated Survival Probability",
        f"{probability:.1%}"
    )

    st.progress(probability)

st.divider()

st.markdown(
    """
    **Project:** Titanic Survival Prediction  
    **Models evaluated:** Logistic Regression, SGD Classifier,
    Decision Tree, Random Forest  
    **Best model:** Tuned Logistic Regression
    """
)
