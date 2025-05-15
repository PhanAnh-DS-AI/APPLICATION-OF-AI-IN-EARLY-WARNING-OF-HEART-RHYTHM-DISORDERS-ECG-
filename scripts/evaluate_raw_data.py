import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_lowpass_data import ECGDataProcessor
from New_model import create_optimized_model
import tensorflow as tf
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(weights_path: str):
    model = create_optimized_model(input_shape=(300, 1))
    model.load_weights(weights_path)
    logger.info(f"Model loaded from {weights_path}")
    return model

def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model performance on test set, with optional inverse scaling."""
    y_pred = model.predict(X_test, verbose=0)
    
    if scaler is not None:
        # Khôi phục về biên độ gốc
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("Evaluation Metrics (on original amplitude):")
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²:   {r2:.4f}")
    
    return y_pred, mse, mae, rmse, r2

def main():
    data_path = "data/raw/data_214_30k.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    weights_path = "./tensorboard_new_model_expv1/logs_customdata/ECG_best_weight/weights-best-epoch-50.weights.h5"

    # 1) Chuẩn bị dữ liệu (đã lọc và chuẩn hóa, scaler được lưu lại)
    processor = ECGDataProcessor(data_path, look_back=look_back, train_split=train_split, val_split=val_split)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, raw_data, filtered_data = processor.process_and_get_data()

    # 2) Nạp mô hình
    model = load_model(weights_path)

    # 3) Đánh giá với inverse transform
    y_pred, mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test, scaler=scaler)

    # 4) Extract original test data for comparison
    # Reconstruct the test segment from filtered_data (unscaled)
    total_length = len(filtered_data)
    train_size = int(total_length * train_split)
    val_size = int(total_length * val_split)
    test_start = train_size + val_size
    test_end = total_length
    original_test_data = filtered_data[test_start:test_end - look_back].flatten()

    # 5) Vẽ biểu đồ so sánh trên biên độ gốc
    plt.figure(figsize=(12, 6))
    plt.plot(original_test_data[1500:2500], label='Tín hiệu gốc (filtered, unscaled)', color='green')
    plt.plot(y_test[1500:2500], label='Tín hiệu thực tế (inverse transformed)', color='blue')
    plt.plot(y_pred[1500:2500], label='Tín hiệu dự đoán (inverse transformed)')#,-Tools for working with X posts and user data are currently unavailable due to API restrictions, but I can still assist with code, data analysis, or other questions! Let me know how I can help.

    plt.title('So sánh tín hiệu ECG gốc, thực tế và tái lập', fontsize=16)
    plt.xlabel('Time steps', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend()
    plt.savefig('reconstructed_ecg_original_amplitude.png')
    plt.show()

    # 6) Lưu kết quả
    with open('evaluation_metrics_original_amplitude.txt', 'w') as f:
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n")
    logger.info("Evaluation metrics on original amplitude saved.")
    
if __name__ == "__main__":
    main()