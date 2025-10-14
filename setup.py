#!/usr/bin/env python3
"""
Setup script for Academic Risk Predictor
This script helps set up the environment and generate necessary files for the application to run.
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python 3.6+ is installed"""
    if sys.version_info < (3, 6):
        print("Python 3.6+ is required. Please upgrade your Python version.")
        return False
    return True

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Required packages installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install required packages. Please check your internet connection and try again.")
        return False

def generate_model():
    """Generate the model using the sample data"""
    # Check if model already exists
    if os.path.exists('student_pipe.pkl'):
        print("Model already exists. Skipping generation.")
        return True
    
    try:
        subprocess.check_call([sys.executable, "generate_model.py"])
        print("Model generated successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to generate model.")
        return False

def main():
    print("Academic Risk Predictor - Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    print("\nInstalling required packages...")
    if not install_requirements():
        sys.exit(1)
    
    # Generate model
    print("\nGenerating model...")
    if not generate_model():
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("\nTo run the application, execute:")
    print("  streamlit run app.py")
    print("\nThen open your browser to http://localhost:8501")

if __name__ == "__main__":
    main()