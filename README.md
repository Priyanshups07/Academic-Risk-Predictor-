# Academic Risk Predictor - ML Model

This project leverages machine learning to assess the risk of academic failure for students based on various personal, academic, and socio-economic factors. The goal is to provide educators and counselors with actionable insights to identify at-risk students and implement timely support strategies.

## Project Overview

The application is built using Python and Streamlit for the web interface. It uses a trained machine learning pipeline to assess academic risk based on user input. The model is trained on a dataset containing features such as attendance, previous marks, health issues, stress level, social media usage, study hours, family support, socio-economic status, and past academic failures.

## Project Structure and File Descriptions

- `app.py`: The main Streamlit web application. It provides an interactive interface for users to input student data and receive risk assessments. The app loads the trained model and dataset, collects user input, and displays the risk prediction and probability scores.
- `retrain_and_save_model.py`: A script used to retrain the machine learning model on the dataset and save the updated model pipeline as a pickle file (`student_pipe.pkl`). Run this script if you want to update the model with new data or change the model parameters.
- `student_success_data.csv`: The dataset containing student records used for training and testing the machine learning model. Each row represents a student and includes features relevant to academic success.
- `student_pipe.pkl`: The serialized (pickled) machine learning pipeline. This file is loaded by `app.py` to make predictions. Ensure this file is present in the project directory before running the app.
- `student_df.pkl`: (If present) A pickled version of the DataFrame, possibly used for quick data loading or backup. Not required for running the app, but may be useful for development.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies**
   Make sure you have Python 3.10 or higher installed. Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Start the Streamlit web app with:
   ```bash
   streamlit run app.py
   ```
   This will open the app in your default web browser. If it does not open automatically, follow the link provided in the terminal (usually http://localhost:8501).

## Usage Guide

1. **Input Student Data**: Use the sliders and dropdowns in the web app to enter the student's information for each feature (attendance, previous marks, health issues, etc.).
2. **Predict**: Click the "Assess Academic Risk" button. The app will display the risk assessment (e.g., "Successful" or "At Risk") and, if available, the probability of each outcome. High risk probability indicates a need for intervention.
3. **Interpret Results**: Use the prediction and probability scores to inform interventions or support strategies for the student.

## Model Retraining

If you want to retrain the model with new or updated data:
1. Update or replace `student_success_data.csv` with your new dataset.
2. Run the retraining script:
   ```bash
   python retrain_and_save_model.py
   ```
3. This will generate a new `student_pipe.pkl` file, which will be used by the app for future predictions.

## Troubleshooting

- **Missing Files**: Ensure `student_pipe.pkl` and `student_success_data.csv` are present in the project directory. The app will not run without these files.
- **Python Version**: The project requires Python 3.10 or higher. Check your version with `python --version`.
- **Dependency Issues**: If you encounter errors related to missing packages, re-run `pip install -r requirements.txt`.
- **Model Errors**: If you see errors about loading the model or making predictions, try retraining the model using the provided script.

## Additional Notes

- The virtual environment (if used) is not included in the repository. It is recommended to create a virtual environment for dependency management:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```
- The app is intended for educational and demonstration purposes. For production use, further validation and security measures are recommended.
- For questions or contributions, please open an issue or submit a pull request.
