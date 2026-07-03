import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("models/churn_model.pkl")

scaler = joblib.load("models/scaler.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/data.csv")
feature_importance = pd.read_csv("data/feature_importance.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---------------- SIDEBAR ----------------
st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "",
    [
        "Home",
        "Dashboard",
        "Predict Customer",
        "Model Performance",
        "About"
    ]
)

# ==================================================
# HOME
# ==================================================
if page == "Home":

    st.title(" Customer Churn Prediction Dashboard")

    st.markdown("""
    

    This application predicts whether a telecom customer is likely to churn using a Machine Learning model.

    **Tech Stack**
    - Python
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Streamlit
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", "7043")
    col2.metric("Churn Rate", "26.5%")
    col3.metric("Accuracy", "80.7%")
    col4.metric("Model", "Logistic Regression")

    st.divider()

    st.subheader("Project Workflow")

    st.write("""
     Data Cleaning

      Exploratory Data Analysis

      Feature Engineering

      Model Training

      Model Evaluation

      Customer Churn Prediction
    """)

# ==================================================
# DASHBOARD
# ==================================================
elif page == "Dashboard":

    st.title(" Dashboard")

    col1, col2 = st.columns(2)

    # ---------------- Churn ----------------
    with col1:

        fig, ax = plt.subplots(figsize=(5,4))

        df["Churn"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Customer Churn")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Customers")

        st.pyplot(fig)

    # ---------------- Contract ----------------
    with col2:

        fig, ax = plt.subplots(figsize=(5,4))

        df["Contract"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Contract Type")
        ax.set_xlabel("Contract")
        ax.set_ylabel("Customers")

        st.pyplot(fig)

    st.divider()

    col3, col4 = st.columns(2)

    # ---------------- Monthly Charges ----------------
    with col3:

        fig, ax = plt.subplots(figsize=(5,4))

        ax.hist(df["MonthlyCharges"], bins=20)

        ax.set_title("Monthly Charges Distribution")
        ax.set_xlabel("Monthly Charges")

        st.pyplot(fig)

    # ---------------- Tenure ----------------
    with col4:

        fig, ax = plt.subplots(figsize=(5,4))

        ax.hist(df["tenure"], bins=20)

        ax.set_title("Tenure Distribution")
        ax.set_xlabel("Months")

        st.pyplot(fig)

    st.divider()

    st.subheader(" Top 10 Features Affecting Churn")

    top10 = feature_importance.head(10)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        top10["Feature"],
        top10["Importance"]
    )

    ax.invert_yaxis()

    ax.set_title("Top 10 Most Important Features")
    ax.set_xlabel("Importance")

    st.pyplot(fig)

    st.markdown("""
    ###  Business Insights

    - Customers with **month-to-month contracts** are more likely to churn.
    - Customers with **short tenure** are at greater risk of leaving.
    - **Higher monthly charges** are associated with increased churn.
    - Long-term contracts can significantly improve customer retention.
    """)

# ==================================================
# PREDICTION
# ==================================================
elif page == "Predict Customer":

    st.title(" Customer Churn Prediction")

    st.write("Enter customer details below.")

    col1, col2 = st.columns(2)

    with col1:

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        MonthlyCharges = st.slider(
            "Monthly Charges",
            18.0,
            120.0,
            70.0
        )

        TotalCharges = st.number_input(
            "Total Charges",
            0.0,
            9000.0,
            1000.0
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        Partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )
        DeviceProtection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        TechSupport = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    with col2:

        Dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        PhoneService = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        Contract = st.selectbox(
            "Contract",
            ["Month-to-month",
             "One year",
             "Two year"]
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        
        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        
        OnlineSecurity = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        
        OnlineBackup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        

    st.divider()
    
    if st.button("Predict Churn"):
    
        
        customer = {
            "SeniorCitizen": SeniorCitizen,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "gender_Male": 1 if gender == "Male" else 0,
            "Partner_Yes": 1 if Partner == "Yes" else 0,
            "Dependents_Yes": 1 if Dependents == "Yes" else 0,
            "PhoneService_Yes": 1 if PhoneService == "Yes" else 0,
            "MultipleLines_No phone service": 1 if MultipleLines == "No phone service" else 0,
            "MultipleLines_Yes": 1 if MultipleLines == "Yes" else 0,
            "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,
            "InternetService_No": 1 if InternetService == "No" else 0,
            "OnlineSecurity_No internet service": 1 if OnlineSecurity == "No internet service" else 0,
            "OnlineSecurity_Yes": 1 if OnlineSecurity == "Yes" else 0,
            "OnlineBackup_No internet service": 1 if OnlineBackup == "No internet service" else 0,
            "OnlineBackup_Yes": 1 if OnlineBackup == "Yes" else 0,
            "DeviceProtection_No internet service": 1 if DeviceProtection == "No internet service" else 0,
            "DeviceProtection_Yes": 1 if DeviceProtection == "Yes" else 0,
            "TechSupport_No internet service": 1 if TechSupport == "No internet service" else 0,
            "TechSupport_Yes": 1 if TechSupport == "Yes" else 0,
            "StreamingTV_No internet service": 1 if StreamingTV == "No internet service" else 0,
            "StreamingTV_Yes": 1 if StreamingTV == "Yes" else 0,
            "StreamingMovies_No internet service": 1 if StreamingMovies == "No internet service" else 0,
            "StreamingMovies_Yes": 1 if StreamingMovies == "Yes" else 0,
            "Contract_One year": 1 if Contract == "One year" else 0,
            "Contract_Two year": 1 if Contract == "Two year" else 0,
            "PaperlessBilling_Yes": 1 if PaperlessBilling == "Yes" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0
        }
        
        input_df = pd.DataFrame([customer])

       
        
        feature_columns = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'gender_Male', 'Partner_Yes', 'Dependents_Yes',
            'PhoneService_Yes', 'MultipleLines_No phone service',
            'MultipleLines_Yes', 'InternetService_Fiber optic',
            'InternetService_No', 'OnlineSecurity_No internet service',
            'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
            'OnlineBackup_Yes', 'DeviceProtection_No internet service',
            'DeviceProtection_Yes', 'TechSupport_No internet service',
            'TechSupport_Yes', 'StreamingTV_No internet service',
            'StreamingTV_Yes', 'StreamingMovies_No internet service',
            'StreamingMovies_Yes', 'Contract_One year',
            'Contract_Two year', 'PaperlessBilling_Yes',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
        ]

        input_df = input_df[feature_columns]
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
        input_df = input_df[feature_columns]

        scaled_data = scaler.transform(input_df)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(scaled_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
         st.error(" Customer is likely to churn.")
        else:
            st.success(" Customer is likely to stay.")

        st.metric("Churn Probability", f"{probability*100:.2f}%")

# ==================================================
# MODEL PERFORMANCE
# ==================================================
elif page == "Model Performance":

    st.title(" Model Performance")

    st.metric("Accuracy", "80.7%")

    st.write("### Logistic Regression Results")

    st.code("""
Accuracy : 80.7%

Precision : 0.67

Recall : 0.52

F1 Score : 0.59
""")

# ==================================================
# ABOUT
# ==================================================
elif page == "About":

    st.title(" About")

    st.write("""
       ### Dataset
      IBM Telco Customer Churn Dataset

       ### Algorithms Compared
      - Logistic Regression
      - Decision Tree
      - Random Forest

      ### Final Model
      Logistic Regression

       ### Libraries Used
      - Pandas
      - NumPy
      - Matplotlib
      - Scikit-learn
      - Streamlit
     
       ### Author
        -Vaishanavee Garg""")  