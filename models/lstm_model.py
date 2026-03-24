import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add parent directory to path to import preprocessing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocessing.preprocess import load_data, preprocess_data

def prepare_lstm_data(X, y, time_steps=24):
    """Reshape data for LSTM (samples, time_steps, features)
    Since this dataset is time-series, we ideally want to predict based on past n steps.
    However, for standard tabular data compatibility in the current pipeline, 
    we often reshape tabular data to (samples, 1, features) if proper sequences aren't available,
    or we can create rolling windows.
    Let's create sequence data assuming the dataframe is sorted by time.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        # We take 'time_steps' rows of features
        X_seq.append(X.iloc[i:(i + time_steps)].values)
        # We predict the target at the sequence end
        y_seq.append(y.iloc[i + time_steps])
        
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(time_steps=24):
    """Trains an LSTM model and saves it."""
    print("Loading data...")
    df = load_data()
    
    # Sort by Date and Time
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time_TD'] = pd.to_timedelta(df['Time'])
    df['Datetime'] = df['Date'] + df['Time_TD']
    df = df.sort_values('Datetime').reset_index(drop=True)
    df = df.drop(columns=['Datetime', 'Time_TD'])
    
    print("Preprocessing data...")
    X, y = preprocess_data(df, is_training=True)
    
    print(f"Preparing sequences of length {time_steps} for LSTM...")
    X_seq, y_seq = prepare_lstm_data(X, y, time_steps)
    
    print("Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Linear activation for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training LSTM model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train using a time-series friendly split (do not shuffle sequences arbitrarily if we want true future validation, but fit with validation_split=0.2 takes the last 20%)
    history = model.fit(
        X_seq, y_seq, 
        epochs=50, 
        batch_size=64,
        validation_split=0.2, # The last 20% in time serves as validation
        callbacks=[early_stop],
        verbose=1,
        shuffle=False # Very important for true TS behavior in training split
    )
    
    print("Evaluating model...")
    preds = model.predict(X_seq)
    preds = preds.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_seq, preds))
    mae = mean_absolute_error(y_seq, preds)
    r2 = r2_score(y_seq, preds)
    
    print(f"LSTM Training Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Save the model
    model_path = os.path.join(BASE_DIR, 'models', 'lstm_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save the time_steps info
    import json
    config_path = os.path.join(BASE_DIR, 'models', 'lstm_config.json')
    with open(config_path, 'w') as f:
        json.dump({'time_steps': time_steps}, f)
        
    return model, {"RMSE": rmse, "MAE": mae, "R² Score": r2}

if __name__ == "__main__":
    train_lstm_model()
