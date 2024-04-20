import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

def preprocess_data(data):
    
    scaler = StandardScaler()
    X = np.arange(len(data)).reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    X_reshape = X_scaled.reshape(-1, 1, 1)
    y_pitch = data['pitch']
    y_roll = data['roll']
    y_yaw = data['yaw']
    return X_reshape, y_pitch, y_roll, y_yaw, scaler

def build_model(input_shape):
    
    model = Sequential([
        Conv1D(32, kernel_size=1, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_values(model, next_indices, scaler):
    next_idx = scaler.transform(next_indices)
    next_idx_cnn = next_idx.reshape(-1, 1, 1)
    predict = model.predict(next_idx_cnn)
    return predict.flatten()

def plot_predictions(next_note_time_stamps, predictions_list, labels, colors):
    
    plt.figure(figsize=(10, 6))
    for predictions, label, color in zip(predictions_list, labels, colors):
        plt.plot(next_note_time_stamps, predictions, label=label, linestyle='--', color=color)
    plt.xlabel('note_time')
    plt.ylabel('Values')
    plt.title('Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'database_latest_1sec.csv'
    data = pd.read_csv(file_path)

    X_cnn, y_pitch, y_roll, y_yaw, scaler = preprocess_data(data)

    model_pitch = build_model((X_cnn.shape[1], X_cnn.shape[2]))
    model_roll = build_model((X_cnn.shape[1], X_cnn.shape[2]))
    model_yaw = build_model((X_cnn.shape[1], X_cnn.shape[2]))


    model_pitch.fit(X_cnn, y_pitch, epochs=50, batch_size=16, verbose=0)
    model_roll.fit(X_cnn, y_roll, epochs=50, batch_size=16, verbose=0)
    model_yaw.fit(X_cnn, y_yaw, epochs=50, batch_size=16, verbose=0)

    next_indices = np.arange(len(data), len(data) + 10).reshape(-1, 1)

    predictions_pitch = predict_values(model_pitch, next_indices, scaler)
    predictions_roll = predict_values(model_roll, next_indices, scaler)
    predictions_yaw = predict_values(model_yaw, next_indices, scaler)

    next_note_time_stamps = np.arange(data['timestamp'].max() + 1, data['timestamp'].max() + 11)

    plot_predictions(next_note_time_stamps, [predictions_pitch, predictions_roll, predictions_yaw], ['Pitch', 'Roll', 'Yaw'], ['blue', 'green', 'red'])

if __name__ == "__main__":
    main()



#Since I have not provided with the indepth knowledge of ML I have implemented the algorithm of predicting 10 sec values using CNN model.
#Some part of the above code has been generated using AI Model platform