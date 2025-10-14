# Simple script to generate a basic model for the Academic Risk Predictor

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the student success data
df = pd.read_csv('student_success_data.csv')

# Define features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Handle OneHotEncoder compatibility
# Use sparse_output for scikit-learn >= 1.2 (including current version 1.6.1)
onehot = OneHotEncoder(drop=None, sparse_output=False)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', onehot, categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Create the full pipeline with a Random Forest classifier
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipe.fit(X_train, y_train)

# Save the trained pipeline
with open('student_pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model generated and saved as student_pipe.pkl")
print(f"Model accuracy on test set: {pipe.score(X_test, y_test):.2%}")