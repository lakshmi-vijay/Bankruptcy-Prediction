import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    # Preprocessing
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
    return df

# Load and preprocess the dataset
data = load_data()

# Define the min and max expenses for reverse scaling
min_expenses = data['expenses'].min()
max_expenses = data['expenses'].max()

# Features and target variable
X = data.drop(columns=['expenses'])
y = data['expenses']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit App
st.title("Insurance Expenses Prediction App")
st.write("This app predicts the insurance expenses based on user inputs.")

# Sidebar for user inputs
st.sidebar.header("Enter Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 64, 30)
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    bmi = st.sidebar.number_input("BMI", 16.0, 53.0, 30.0, step=0.1)
    children = st.sidebar.slider("Number of Children", 0, 5, 1)
    smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
    region = st.sidebar.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])
    
    # Map user inputs to numerical values
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    region = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}[region]
    
    # Create a DataFrame with user input
    return pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

# Collect user input
input_df = user_input_features()

# Scale user input
input_scaled = scaler.transform(input_df)

# Add a Predict button
if st.sidebar.button("Predict"):
    # Make predictions
    scaled_prediction = model.predict(input_scaled)
    original_prediction = scaled_prediction[0] * (max_expenses - min_expenses) + min_expenses
    
    # Display results
    st.subheader("Predicted Insurance Expense")
    st.write(f"Estimated Expenses: **${original_prediction:,.2f}**")
