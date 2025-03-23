# evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_data import ECGDataProcessor
from model import create_model
from New_model import create_optimized_model
import tensorflow as tf
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(weights_path: str):
    """Load the trained model with best weights"""
    try:
        model = create_optimized_model(input_shape=(300, 1))  # Giả sử look_back = 300
        model.load_weights(weights_path)
        logger.info(f"Model loaded from {weights_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    try:
        # Dự đoán trên tập test
        y_pred = model.predict(X_test, verbose=0)
        
        # Tính các metric
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        return y_pred, mse, mae, rmse, r2
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise
    
def generate_ecg_signal(model, initial_input, n_steps:int):
    """
    Dự đoán tiếp tín hiệu ECG theo từng bước, thay vì dự đoán toàn bộ tập test một lần.
    
    - model: mô hình đã huấn luyện
    - initial_input: đoạn tín hiệu ban đầu (shape=(1, 300, 1))
    - n_steps: số bước thời gian cần dự đoán
    """
    generated_signal = []
    current_input = initial_input

    for _ in range(n_steps):
        # Dự đoán điểm tiếp theo
        next_point = model.predict(current_input, verbose=0)

        # Lưu giá trị dự đoán
        generated_signal.append(next_point[0, 0])  # Lấy giá trị thực từ array

        # Cập nhật đầu vào: bỏ điểm đầu, thêm điểm mới
        current_input = np.roll(current_input, shift=-1, axis=1)  # Dịch sang trái
        current_input[0, -1, 0] = next_point[0, 0]  # Thêm giá trị mới

    return np.array(generated_signal)

def visualize_predictions(y_true, y_pred):
    """Visualize a single plot comparing actual vs predicted ECG signals"""
    try:
        plt.figure(figsize=(12, 6))  # Chỉnh kích thước hình
        
        # Lấy 300 điểm đầu tiên của tập dữ liệu để vẽ
        plt.plot(y_true[1500:6000], label='Actual Signal', color='blue')
        plt.plot(y_pred[1500:6000], label='Predicted Signal', color='red', linestyle='--')

        # Định dạng biểu đồ
        plt.title('Actual vs Predicted ECG Signal (300 data points)')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Lưu và hiển thị hình ảnh
        # plt.savefig('output_figure.png')
        plt.show()
        
        logger.info("Predictions visualized and saved as output_figure.png")
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        raise

def visualize_predictions_generated(y_true, y_pred, y_generated):
    """Visualize actual vs predicted vs generated ECG signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:600], label='Actual Signal', color='blue')
    plt.plot(y_pred[:600], label='Predicted Signal (Direct)', color='red', linestyle='--')
    plt.plot(y_generated, label='Generated Signal (Step-by-Step)', color='green', linestyle='dotted')
    plt.title('ECG Signal Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    logger.info("Predictions visualized successfully.")

def main():
    # Parameters data/processed/data_102_filtered_100k.csv | data\processed\data_214_30k_filtered.csv
    data_path = "./data/processed/data_214_30k_filtered.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    weights_path = "./logs/ECG_best_weight/weights-best-epoch-40.weights.h5"  # Đường dẫn đến weights tốt nhất

    try:
        # Initialize data processor
        processor = ECGDataProcessor(data_path, look_back, train_split, val_split)
        logger.info("Loading and processing ECG data for evaluation...")
        X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

        # Load trained model
        model = load_model(weights_path)

        # Evaluate model
        y_pred, mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Visualize predictions
        # visualize_predictions(y_test, y_pred )
        
        y_generated = generate_ecg_signal(model, X_test[:1], n_steps=600)
        visualize_predictions_generated(y_test, y_pred, y_generated)


        # Save evaluation metrics to file (optional)
        with open('evaluation_metrics.txt', 'w') as f:
            f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")
        logger.info("Evaluation metrics saved to evaluation_metrics.txt")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()