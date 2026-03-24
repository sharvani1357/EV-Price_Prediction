import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models dir if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(file_name="Charging station_A_Calif.csv"):
    """Load dataset using pandas from data folder"""
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Handle missing values, duplicate rows, and outliers"""
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values (forward fill then backward fill for time series, or mean for numerical)
    # We will use ffill and bfill since it's time series data
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Alternatively simple imputation
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Simple outlier handling (capping at 1st and 99th percentile for numeric columns)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
        
    return df

def feature_engineering(df):
    """Add peak hour feature, extract DayType from Date"""
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract DayType (Weekend / Weekday)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayType'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Parse time
    def extract_hour(time_str):
        try:
            return int(str(time_str).split(':')[0])
        except:
            return 0
            
    df['Hour'] = df['Time'].apply(extract_hour)
    
    # Add Peak Hour feature (assuming 7-10 and 17-20 are peak hours)
    df['IsPeakHour'] = df['Hour'].apply(lambda h: 1 if (7 <= h <= 10) or (17 <= h <= 20) else 0)
    
    return df

def preprocess_data(df, is_training=True):
    """Convert Time to numerical, encode categorical, scale numerical."""
    df = df.copy()
    
    if is_training:
        df = clean_data(df)
        df = feature_engineering(df)
    
    # Time encoding (Sine/Cosine for Hour)
    if 'Hour' in df.columns:
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        
    # Categorical variables encoding
    cat_cols = ['Weather Conditions', 'DayType', 'Grid Availability']
    
    # We will use LabelEncoder and save them
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                joblib.dump(le, os.path.join(MODELS_DIR, f'le_{col}.pkl'))
            else:
                # Handle unseen categories
                le_path = os.path.join(MODELS_DIR, f'le_{col}.pkl')
                if os.path.exists(le_path):
                    le = joblib.load(le_path)
                    # For unknown classes, map to a known class or standard encoding (like -1)
                    # We map them to the first class to prevent crashing
                    known_classes = list(le.classes_)
                    df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else known_classes[0])
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0

    # Define numerical features to scale
    target_col = 'EV Charging Demand (kW)'
    
    # Features to keep for training (excluding Date, Time)
    features = [c for c in df.columns if c not in ['Date', 'Time', 'DayOfWeek', 'Hour', target_col]]
    
    if is_training:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    else:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            # Ensure all features exist in df
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            df[features] = scaler.transform(df[features])
            
    # For DL models, ensuring uniform data types
    df[features] = df[features].astype('float32')
    
    if is_training and target_col in df.columns:
        df[target_col] = df[target_col].astype('float32')
        return df[features], df[target_col]
        
    return df[features]
