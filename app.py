import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("loan_data.csv")

le = LabelEncoder()
data["person_gender"] = le.fit_transform(data["person_gender"])
data["person_education"] = le.fit_transform(data["person_education"])
data["person_home_ownership"] = le.fit_transform(data["person_home_ownership"])
data["loan_intent"] = le.fit_transform(data["loan_intent"])
data["previous_loan_defaults_on_file"] = le.fit_transform(
    data["previous_loan_defaults_on_file"]
)

X = data.drop(columns="loan_status")
y = data["loan_status"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = DecisionTreeClassifier(max_depth=6, min_samples_split=4)
model.fit(x_train, y_train)

import streamlit as st

# ---- SESSION STATE ----
if "started" not in st.session_state:
    st.session_state.started = False

# ---- UI ----
st.title("Welcome To PredLoan!")
st.subheader("Let's predict your chances of getting a loan 🚀")

if st.button("Get Started"):
    st.session_state.started = True

# =============================
# INPUT FORM
# =============================
if st.session_state.started:

    st.subheader("Enter your loan details")

    age = st.number_input("Age", min_value=18, max_value=75, value=25)

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )
    gender = 1 if gender == "Male" else 0

    edu = st.radio(
        "Highest Education",
        ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )
    edu_map = {
        "Associate": 0,
        "Bachelor": 1,
        "Doctorate": 2,
        "High School": 3,
        "Master": 4
    }
    edu = edu_map[edu]

    income = st.number_input("Annual Income", min_value=0.0)

    experience = st.number_input(
        "Years of Employment Experience", min_value=0
    )

    home = st.selectbox(
        "Home Ownership",
        ["MORTGAGE", "OTHER", "OWN", "RENT"]
    )
    home_map = {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    }
    home = home_map[home]

    loan_amt = st.number_input("Loan Amount", min_value=0.0)

    intent = st.selectbox(
        "Loan Intent",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
        ],
    )
    intent_map = {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5,
    }
    intent = intent_map[intent]

    int_rate = st.number_input("Interest Rate (%)", min_value=0.0)

    percent_income = st.number_input(
        "Loan % of Income", min_value=0.0
    )

    credit_years = st.number_input(
        "Credit History (Years)", min_value=0.0
    )

    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900
    )

    prev_loan = st.radio(
        "Previous Loan Defaults?",
        ["No", "Yes"]
    )
    prev_loan = 1 if prev_loan == "Yes" else 0

    # =============================
    # PREDICTION
    # =============================
    if st.button("Predict Loan Status"):

        if income <= 0 or loan_amt <= 0:
            st.error("Income and Loan Amount must be greater than zero.")
            st.stop()

        x_input = np.array([[
            age, gender, edu, income, experience,
            home, loan_amt, intent,
            int_rate, percent_income,
            credit_years, credit_score,
            prev_loan
        ]])

        x_input = scaler.transform(x_input)
        prediction = model.predict(x_input)

        if prediction[0] == 1:
            st.success("✅ High probability of getting the loan!")
        else:
            st.error("❌ Low probability of getting the loan.")

st.write("Thank you for choosing PredLoan ❤️")
