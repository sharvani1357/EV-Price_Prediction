import os
import json
import numpy as np

# Suppress TensorFlow logging and disable GPU to prevent init errors if strictly running lightweight
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Uncomment if forcing CPU: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from preprocessing.preprocess import preprocess_data
from models.linear_model import train_linear_model
from models.ann_model import train_ann_model
from models.lstm_model import train_lstm_model

# Add parent directory to path to import preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_all_models():
    """Train all models, evaluate, and save the best one."""
    print("="*50)
    print("Starting Model Training Pipeline")
    print("="*50)
    
    # 1. Train Linear Regression
    print("\n[1/3] Training Linear Regression Model...")
    lr_model, lr_metrics = train_linear_model()
    
    # 2. Train ANN
    print("\n[2/3] Training Artificial Neural Network (ANN)...")
    ann_model, ann_metrics = train_ann_model()
    
    # 3. Train LSTM
    print("\n[3/3] Training Long Short-Term Memory (LSTM) Network...")
    lstm_model, lstm_metrics = train_lstm_model(time_steps=24) # 24 hours lookback
    
    print("\n" + "="*50)
    print("Model Evaluation Summary")
    print("="*50)
    
    results = {
        "Linear Regression": lr_metrics,
        "ANN": ann_metrics,
        "LSTM": lstm_metrics
    }
    
    # Print comparison
    print(f"{'Model':<20} | {'RMSE':<10} | {'MAE':<10} | {'R² Score':<10}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | {metrics['RMSE']:<10.4f} | {metrics['MAE']:<10.4f} | {metrics['R² Score']:<10.4f}")
        
    print("\n" + "="*50)
    
    # Select the best model based on highest R² Score (or lowest RMSE/MAE)
    best_model_name = max(results, key=lambda x: results[x]['R² Score'])
    best_metrics = results[best_model_name]
    
    print(f"🌟 Best Model: {best_model_name} 🌟")
    print(f"Metrics: R² = {best_metrics['R² Score']:.4f}, RMSE = {best_metrics['RMSE']:.4f}")
    
    # Save the best model configuration for inference
    best_config = {
        "best_model_name": best_model_name,
        "metrics": best_metrics
    }
    
    config_path = os.path.join(BASE_DIR, 'models', 'best_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
        
    print(f"Best model configuration saved to {config_path}")

if __name__ == "__main__":
    train_all_models()
