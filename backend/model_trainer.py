import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

def build_model(model_type='LSTM', units=50, input_shape=(None, 1)):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units, input_shape=input_shape))
    elif model_type == 'RNN':
        model.add(SimpleRNN(units, input_shape=input_shape))
    else:
        raise ValueError("model_type must be 'LSTM' or 'RNN'")
        
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_and_train_model(df, model_type='LSTM', epochs=10, batch_size=32, test_split=0.2, seq_length=60):
    # Take the dataframe, scale features with StandardScaler
    scaler = StandardScaler()
    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_prices)
    
    # Split 3D temporal arrays (samples, timesteps, 1)
    X, y = create_sequences(scaled_close, seq_length)
    
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # build model
    model = build_model(model_type, units=50, input_shape=(seq_length, 1))
    
    # train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )
    
    # predict on X_test
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # inverse transform to get actual prices
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)
    
    # calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    
    return {
        'history': history.history,
        'predictions': y_pred_actual.flatten(),
        'actuals': y_test_actual.flatten(),
        'rmse': rmse,
        'mae': mae
    }
