# Import necessary libraries for data manipulation, model building, and evaluation
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import sys

# Try importing XGBoost, set to None if not available
try:
    from xgboost import XGBClassifier 
    print("XGBoost is installed.")
except ImportError:
    XGBClassifier = None
    print("Warning: XGBoost is not installed. XGBoost model will be skipped.")

# Try importing SMOTE for class balancing, set to None if not available
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
    print("Warning: imbalanced-learn (SMOTE) is not installed. Class balancing will be skipped if needed.")

# Load the student success data from CSV with error handling
try:
    df = pd.read_csv('student_success_data.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Define the target column and split features and target
# 'Target' is the column to predict (e.g., 'Successful', 'At Risk')
target_col = 'Target'
X = df.drop(columns=[target_col])  # Features (all columns except target)
y = df[target_col]                 # Target variable

# Encode target labels for compatibility with all models
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_map = dict(zip(le.classes_, le.transform(le.classes_)))  # Map class names to integers
inv_label_map = {v: k for k, v in label_map.items()}           # Map integers back to class names

# Identify categorical and numerical features for preprocessing
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Handle OneHotEncoder compatibility for different sklearn versions
onehot_kwargs = {}
try:
    # Try using sparse_output (sklearn >=1.2)
    onehot = OneHotEncoder(drop=None, sparse_output=False)
except TypeError:
    # Fallback for older sklearn versions
    onehot = OneHotEncoder(drop=None, sparse=False)

# Create a preprocessing pipeline: OneHotEncode categorical, scale numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', onehot, categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Balance classes using SMOTE if class imbalance is detected and SMOTE is available
imbalance_ratio = y.value_counts().min() / y.value_counts().max() if y.value_counts().max() > 0 else 1
if imbalance_ratio < 0.8:
    if SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
        print('SMOTE applied for class balancing.')
    else:
        print('Warning: Class imbalance detected but SMOTE is not available. Proceeding without balancing.')
        X_bal, y_bal = X, y
else:
    X_bal, y_bal = X, y

# For XGBoost, encoded labels are needed; for others, string labels are fine
# Prepare encoded labels for XGBoost compatibility
if XGBClassifier is not None:
    y_bal_encoded = le.transform(y_bal)

# Split the balanced data into training and test sets (stratified)
try:
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.15, random_state=2, stratify=y_bal)
    if XGBClassifier is not None:
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
except Exception as e:
    print(f"Error during train/test split: {e}")
    sys.exit(1)

# Define a list to hold models and their parameter grids for hyperparameter tuning
models = []

# Add Random Forest model and its parameter grid
models.append(('RandomForest', RandomForestClassifier(random_state=42), {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__class_weight': ['balanced']
}))
# Add Gradient Boosting model and its parameter grid
models.append(('GradientBoosting', GradientBoostingClassifier(random_state=42), {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.1, 0.2],
    'classifier__max_depth': [3, 5]
}))
# Add XGBoost model and its parameter grid if XGBoost is available
if XGBClassifier is not None:
    models.append(('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.1, 0.2],
        'classifier__max_depth': [3, 5]
    }))
# Add Logistic Regression model and its parameter grid
models.append(('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42), {
    'classifier__C': [0.1, 1, 10],
    'classifier__class_weight': ['balanced']
}))

# Initialize variables to track the best model and its performance
best_score = 0
best_model = None
best_name = ''
best_params = None

# Loop through each model, perform GridSearchCV, and select the best model
for name, clf, param_grid in models:
    # Create a pipeline with preprocessing and classifier
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    print(f'\nTuning {name}...')
    # For XGBoost, use encoded labels; for others, use original labels
    if name == 'XGBoost':
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train_encoded)
        score = grid.best_score_
    else:
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        score = grid.best_score_
    print(f'{name} best CV accuracy: {score*100:.2f}%')
    # Update best model if current model is better
    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_name = name
        best_params = grid.best_params_

# Evaluate the best model on the test set
try:
    if best_name == 'XGBoost':
        y_pred = best_model.predict(X_test)
        y_pred = [inv_label_map[i] for i in y_pred]  # Convert encoded labels back to original
        test_acc = accuracy_score(y_test, y_pred)
        print(f'\nBest model: {best_name}')
        print(f'Best parameters: {best_params}')
        print(f'Test set accuracy: {test_acc*100:.2f}%')
        print('\nClassification report:\n', classification_report(y_test, y_pred))
    else:
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f'\nBest model: {best_name}')
        print(f'Best parameters: {best_params}')
        print(f'Test set accuracy: {test_acc*100:.2f}%')
        print('\nClassification report:\n', classification_report(y_test, y_pred))
except Exception as e:
    print(f"Error during model evaluation: {e}")

# Save the original DataFrame and the best model pipeline to disk
try:
    with open('student_df.pkl', 'wb') as f:
        pickle.dump(df, f)  # Save the DataFrame
    with open('student_pipe.pkl', 'wb') as f:
        pickle.dump(best_model, f)  # Save the trained pipeline
    print('Retraining and saving completed for student success data.')
except Exception as e:
    print(f"Error saving model or data: {e}")
    sys.exit(1) 