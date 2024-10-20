import pandas as pd
import pickle 
import streamlit as st

# Load the trained model
model = pickle.load(open('G:/DataScience/AIcourse/COPD Prediction/Prediction/Best_Decision_Tree_Model.pkl', 'rb'))

# Streamlit app
def main():
    st.title("COPD Prediction Dashboard")

    # User input
    st.sidebar.header("User Input")
    age = st.sidebar.slider("Age", 30, 80, 50)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    bmi = st.sidebar.slider("BMI", 10, 40, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
    biomass_fuel_exposure = st.sidebar.selectbox("Biomass Fuel Exposure", ["Yes", "No"])
    occupational_exposure = st.sidebar.selectbox("Occupational Exposure", ["Yes", "No"])
    family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
    air_pollution_level = st.sidebar.slider("Air Pollution Level", 0, 300, 50)
    respiratory_infections = st.sidebar.selectbox("Respiratory Infections in Childhood", ["Yes", "No"])
    location = st.sidebar.selectbox("Location", ["Kathmandu", "Pokhara", "Biratnagar", "Lalitpur", "Birgunj", "Chitwan", "Hetauda", "Dharan", "Butwal"])

    # Process the input data
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Biomass_Fuel_Exposure': [biomass_fuel_exposure],
        'Occupational_Exposure': [occupational_exposure],
        'Family_History_COPD': [family_history],
        'BMI': [bmi],
        'Air_Pollution_Level': [air_pollution_level],
        'Respiratory_Infections_Childhood': [respiratory_infections],
        'Smoking_Status_encoded': [smoking_status],
        'Location': [location]  # Add Location here properly
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Encoding
    input_df['Gender'] = input_df["Gender"].map({'Male': 1, 'Female': 0})
    input_df['Smoking_Status_encoded'] = input_df['Smoking_Status_encoded'].map({'Current': 1, 'Former': 0.5, 'Never': 0})
    input_df['Biomass_Fuel_Exposure'] = input_df["Biomass_Fuel_Exposure"].map({'Yes': 1, 'No': 0})
    input_df['Occupational_Exposure'] = input_df["Occupational_Exposure"].map({'Yes': 1, 'No': 0})
    input_df['Family_History_COPD'] = input_df["Family_History_COPD"].map({'Yes': 1, 'No': 0})
    input_df['Respiratory_Infections_Childhood'] = input_df["Respiratory_Infections_Childhood"].map({'Yes': 1, 'No': 0})

    # One-hot encode 'Location'
    if 'Location' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['Location'], drop_first=True)
    else:
        st.error("Location column is missing in the input data.")

    # Ensure the input DataFrame matches model's expected columns
    # You might need X_train to compare columns if you have a specific set of columns used during training
    # for col in set(X_train.columns) - set(input_df.columns):
    #     input_df[col] = 0  # Add missing columns with zeros
    # input_df = input_df[X_train.columns]  # Ensure correct column order

    # Predictions
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write("COPD is Detected")
    else:
        st.write("COPD is not Detected")

if __name__ == "__main__":
    main()


