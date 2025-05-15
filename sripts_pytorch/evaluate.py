import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_data import ECGDataProcessor
from model_grid_search import model_grid_optimize, SMAPE
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(weights_path: str):
    """Load the trained model with best weights"""
    try:
        model = model_grid_optimize(input_shape=(300, 1))  # Giả sử look_back = 300
        model.load_state_dict(torch.load(weights_path))  # Load weights PyTorch
        model.eval()  # Chuyển mô hình sang chế độ evaluation
        logger.info(f"Model loaded from {weights_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    try:
        # Chuyển đổi dữ liệu numpy sang tensor của PyTorch
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Dự đoán trên tập test
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy().flatten()  # Dự đoán bằng PyTorch model

        # Tính các metric
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Tính SMAPE
        smape = SMAPE()
        y_true_tensor = torch.tensor(y_test, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        smape_score = smape(y_true_tensor, y_pred_tensor).item()  # Chuyển từ tensor sang giá trị Python

        logger.info(f"Evaluation Metrics:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"SMAPE: {smape_score:.4f}")

        return y_pred, mse, mae, rmse, r2, smape_score
        
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
    current_input = torch.tensor(initial_input, dtype=torch.float32)

    for _ in range(n_steps):
        # Dự đoán điểm tiếp theo
        with torch.no_grad():
            next_point = model(current_input).numpy()  # Dự đoán với mô hình PyTorch

        # Lưu giá trị dự đoán
        generated_signal.append(next_point[0, 0])

        # Cập nhật đầu vào: bỏ điểm đầu, thêm điểm mới
        current_input = torch.roll(current_input, shifts=-1, dims=1)
        current_input[0, -1, 0] = next_point[0, 0]  # Thêm giá trị mới

    return np.array(generated_signal)

def visualize_predictions(y_true, y_pred):
    """Visualize a single plot comparing actual vs predicted ECG signals"""
    try:
        plt.figure(figsize=(12, 6))  # Chỉnh kích thước hình
        
        # Lấy 300 điểm đầu tiên của tập dữ liệu để vẽ
        plt.plot(y_true[7000:10000], label='Tín hiệu thực tế', color='blue')
        plt.plot(y_pred[7000:10000], label='Tín hiệu dự đoán', color='red', linestyle='--')

        # Định dạng biểu đồ
        plt.title('Tín hiệu ECG thực tế và dư đoán (Model Custom Data) - Thử Nghiệm 1',fontsize=18)
        plt.xlabel('Chuỗi thời gian', fontsize=16)
        plt.ylabel('Biên độ',fontsize=16)
        plt.legend()
        
        # Lưu và hiển thị hình ảnh
        plt.show()
        
        logger.info("Predictions visualized successfully.")
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        raise

def visualize_predictions_generated(y_true, y_pred, y_generated):
    """Visualize actual vs predicted vs generated ECG signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[12500:], label='Actual Signal', color='blue')
    plt.plot(y_pred[12500:], label='Predicted Signal (Direct)', color='red', linestyle='--')
    plt.plot(y_generated, label='Generated Signal (Step-by-Step)', color='green', linestyle='dotted')
    plt.title('ECG Signal Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    logger.info("Predictions visualized successfully.")

def main():
    data_path = "data/processed/100_Test1.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    weights_path = "./tensorboard_new_model_expv3/logs_orginaldata/ECG_best_weight/weights-best-epoch-60.weights.pth"  # Đường dẫn đến weights tốt nhất

    try:
        # Initialize data processor
        processor = ECGDataProcessor(data_path, look_back, train_split, val_split)
        logger.info("Loading and processing ECG data for evaluation...")
        X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

        # Load trained model
        model = load_model(weights_path)

        # Evaluate model
        y_pred, mse, mae, rmse, r2, smape_score = evaluate_model(model, X_test, y_test)

        # Visualize predictions
        visualize_predictions(y_test, y_pred)
        
        # Generate ECG signal
        y_generated = generate_ecg_signal(model, X_test[:1], n_steps=600)
        visualize_predictions_generated(y_test, y_pred, y_generated)

        # Save evaluation metrics to file
        with open('evaluation_metrics.txt', 'w') as f:
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"SMAPE: {smape_score:.4f}\n")
        logger.info("Evaluation metrics saved to evaluation_metrics.txt")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

