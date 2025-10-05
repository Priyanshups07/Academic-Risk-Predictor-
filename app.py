# Import the Streamlit library for building the web app interface
import streamlit as st
# Import pickle for loading the trained machine learning model
import pickle
# Import numpy for numerical operations (not used directly here, but often used with ML models)
import numpy as np
# Import pandas for data manipulation and reading CSV files
import pandas as pd

# Set the title of the Streamlit web app
st.title("Academic Risk Predictor")

# Load the trained machine learning pipeline (model) with error handling
try:
    # Attempt to load the model pipeline from the pickle file
    pipe = pickle.load(open('student_pipe.pkl', 'rb'))
except Exception as e:
    # If loading fails, display an error message and stop the app
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the student data CSV file with error handling
try:
    # Attempt to read the CSV file into a pandas DataFrame
    df = pd.read_csv('student_success_data.csv')
except Exception as e:
    # If loading fails, display an error message and stop the app
    st.error(f"Error loading data: {e}")
    st.stop()

# Create a slider for Attendance percentage, using min, max, and mean from the data
attendance = st.slider(
    'Attendance (%)',
    int(df['Attendance'].min()),
    int(df['Attendance'].max()),
    int(df['Attendance'].mean())
)

# Create a slider for Previous Marks percentage
previous_marks = st.slider(
    'Previous Marks (%)',
    int(df['Previous_marks'].min()),
    int(df['Previous_marks'].max()),
    int(df['Previous_marks'].mean())
)

# Create a dropdown (selectbox) for Health Issues, using unique values from the data
health_issues = st.selectbox(
    'Health Issues',
    df['Health_issues'].unique()
)

# Create a slider for Stress Level (scale 1-10)
stress_level = st.slider(
    'Stress Level (1-10)',
    int(df['Stress_level'].min()),
    int(df['Stress_level'].max()),
    int(df['Stress_level'].mean())
)

# Create a slider for Social Media Usage (hours per day)
social_media_usage = st.slider(
    'Social Media Usage (hours/day)',
    int(df['Social_media_usage'].min()),
    int(df['Social_media_usage'].max()),
    int(df['Social_media_usage'].mean())
)

# Create a slider for Study Hours (hours per day)
study_hours = st.slider(
    'Study Hours (hours/day)',
    int(df['Study_hours'].min()),
    int(df['Study_hours'].max()),
    int(df['Study_hours'].mean())
)

# Create a dropdown (selectbox) for Family Support, using unique values from the data
family_support = st.selectbox(
    'Family Support',
    df['Family_support'].unique()
)

# Create a dropdown (selectbox) for Socio Economic Status, using unique values from the data
socio_economic_status = st.selectbox(
    'Socio Economic Status',
    df['Socio_economic_status'].unique()
)

# Create a slider for Past Academic Failures
past_academic_failures = st.slider(
    'Past Academic Failures',
    int(df['Past_academic_failures'].min()),
    int(df['Past_academic_failures'].max()),
    int(df['Past_academic_failures'].mean())
)

# When the user clicks the 'Assess Academic Risk' button, run the prediction
if st.button('Assess Academic Risk'):
    # Prepare the input data as a DataFrame for the model
    query_df = pd.DataFrame({
        'Attendance': [attendance],
        'Previous_marks': [previous_marks],
        'Health_issues': [health_issues],
        'Stress_level': [stress_level],
        'Social_media_usage': [social_media_usage],
        'Study_hours': [study_hours],
        'Family_support': [family_support],
        'Socio_economic_status': [socio_economic_status],
        'Past_academic_failures': [past_academic_failures]
    })
    try:
        # Make a prediction using the loaded model pipeline
        prediction = pipe.predict(query_df)[0]
        # If the model supports probability estimates, display them
        if hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(query_df)[0]
            # Get the class labels from the model
            class_labels = pipe.classes_
            # Map each class label to its predicted probability
            proba_dict = {label: p for label, p in zip(class_labels, proba)}
            # Get the probability for 'Successful' and 'At Risk' classes (if present)
            success_percent = int(round(proba_dict.get('Successful', 0) * 100))
            risk_percent = int(round(proba_dict.get('At Risk', 0) * 100))
            # Display the predicted class
            st.success(f"The predicted student status is: {prediction}")
            # Display the probabilities for each class
            st.info(f"Success Probability: {success_percent}%\nRisk Probability: {risk_percent}%")
            
            # Focus on risk assessment
            if prediction == "At Risk":
                st.warning("⚠️ This student is at risk of academic failure. Consider implementing support strategies.")
            else:
                st.success("✅ This student is not at risk of academic failure.")
        else:
            # If probability estimates are not available, just show the prediction
            st.success(f"The predicted student status is: {prediction}")
            st.warning("Probability estimates are not available for this model.")
            
            # Focus on risk assessment
            if prediction == "At Risk":
                st.warning("⚠️ This student is at risk of academic failure. Consider implementing support strategies.")
            else:
                st.success("✅ This student is not at risk of academic failure.")
    except Exception as e:
        # If prediction fails, display an error message
        st.error(f"Prediction error: {e}")

