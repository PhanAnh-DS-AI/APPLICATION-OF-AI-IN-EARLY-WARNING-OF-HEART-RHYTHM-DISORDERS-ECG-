# evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_data import ECGDataProcessor
from scripts.model_grid_search import create_model
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
    
def visualize_predictions(y_true, y_pred, title, save_path):
    """Visualize a single plot comparing actual vs predicted ECG signals"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Lấy 300 điểm để vẽ (hoặc tùy chỉnh nếu cần)
        plt.plot(y_true[1800:], label='Tín hiệu thực tế', color='blue')
        plt.plot(y_pred[1800:], label='Tín hiệu dự đoán', color='red', linestyle='--')

        # Định dạng biểu đồ
        plt.title(title)
        plt.xlabel('Chuỗi thời gian')
        plt.ylabel('Biên độ')
        plt.legend()
        
        # Lưu hình ảnh
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Predictions visualized and saved as {save_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        raise

def main():
    # Parameters
    data_path_214 = "./data/processed/data_214_30k_filtered.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    weights_path_original = "./logs/ECG_best_weight/weights-best-epoch-40.weights.h5"  # Model trên tập gốc
    weights_path_custom = "./logs_customdata/ECG_best_weight/weights-best-epoch-50.weights.h5"  # Model trên tập tùy chỉnh

    try:
        # Initialize data processor for record 214
        processor = ECGDataProcessor(data_path_214, look_back, train_split, val_split)
        logger.info("Loading and processing ECG data for record 214...")
        X_train, y_train, X_val, y_val, X_214, y_214 = processor.process_and_get_data()

        # Load trained models
        model_original = load_model(weights_path_original)
        model_custom = load_model(weights_path_custom)

        # Evaluate model trained on original data
        logger.info("Evaluating model trained on original data on record 214...")
        y_pred_original, mse_orig, mae_orig, rmse_orig, r2_orig = evaluate_model(model_original, X_214, y_214)
        visualize_predictions(y_214, y_pred_original, 
                           'Tín hiệu ECG thực tế và dự đoán trên bản ghi 214 (Model Original Data)', 
                           'reconstructed_214_original.png')

        # Evaluate model trained on custom data
        logger.info("Evaluating model trained on custom data on record 214...")
        y_pred_custom, mse_cust, mae_cust, rmse_cust, r2_cust = evaluate_model(model_custom, X_214, y_214)
        visualize_predictions(y_214, y_pred_custom, 
                           'Tín hiệu ECG thực tế và dự đoán trên bản ghi 214 (Model Custom Data)', 
                           'reconstructed_214_custom.png')

        # Save evaluation metrics to file
        with open('evaluation_metrics_214.txt', 'w') as f:
            f.write("Model trained on original data:\n")
            f.write(f"MSE: {mse_orig:.4f}\nMAE: {mae_orig:.4f}\nRMSE: {rmse_orig:.4f}\nR²: {r2_orig:.4f}\n\n")
            f.write("Model trained on custom data:\n")
            f.write(f"MSE: {mse_cust:.4f}\nMAE: {mae_cust:.4f}\nRMSE: {rmse_cust:.4f}\nR²: {r2_cust:.4f}")
        logger.info("Evaluation metrics for record 214 saved to evaluation_metrics_214.txt")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()