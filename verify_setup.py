#!/usr/bin/env python3
"""
Verification script for Academic Risk Predictor
This script checks if all required files are present and the application can run correctly.
"""

import os
import sys

def check_files():
    """Check if all required files are present"""
    required_files = [
        'app.py',
        'requirements.txt',
        'student_success_data.csv',
        'student_pipe.pkl',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files are present")
        return True

def check_dependencies():
    """Check if required dependencies can be imported"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_model():
    """Check if the model can be loaded"""
    try:
        import pickle
        with open('student_pipe.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def check_data():
    """Check if the data can be loaded"""
    try:
        import pandas as pd
        df = pd.read_csv('student_success_data.csv')
        print(f"✅ Data loaded successfully ({len(df)} records)")
        return True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

def main():
    print("Academic Risk Predictor - Verification Script")
    print("=" * 45)
    
    checks = [
        ("File Check", check_files),
        ("Dependency Check", check_dependencies),
        ("Data Check", check_data),
        ("Model Check", check_model)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 All checks passed! The application is ready to run.")
        print("\nTo run the application, execute:")
        print("  streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")

if __name__ == "__main__":
    main()