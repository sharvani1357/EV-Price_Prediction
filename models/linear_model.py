import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add parent directory to path to import preprocessing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocessing.preprocess import load_data, preprocess_data

def train_linear_model():
    """Trains a linear regression model and saves it."""
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y = preprocess_data(df, is_training=True)
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate on training data (just for sanity check)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"Linear Regression Training Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Save the model
    model_path = os.path.join(BASE_DIR, 'models', 'linear_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, {"RMSE": rmse, "MAE": mae, "R² Score": r2}

if __name__ == "__main__":
    train_linear_model()
