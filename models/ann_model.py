import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add parent directory to path to import preprocessing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocessing.preprocess import load_data, preprocess_data

def train_ann_model():
    """Trains an Artificial Neural Network model and saves it."""
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y = preprocess_data(df, is_training=True)
    
    print("Building ANN model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1) # Linear activation for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training ANN model...")
    # Using early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # We will use a simple 80-20 validation split
    history = model.fit(
        X, y, 
        epochs=100, 
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate on all data (for simplicity in this pipeline)
    print("Evaluating model...")
    preds = model.predict(X)
    
    # Ensure shapes match for sklearn metrics
    preds = preds.flatten()
    
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"ANN Training Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Save the model
    model_path = os.path.join(BASE_DIR, 'models', 'ann_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, {"RMSE": rmse, "MAE": mae, "R² Score": r2}

if __name__ == "__main__":
    train_ann_model()
