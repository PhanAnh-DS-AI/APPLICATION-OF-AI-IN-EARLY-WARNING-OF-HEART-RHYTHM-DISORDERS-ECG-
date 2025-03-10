# evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_data import ECGDataProcessor
from model import create_model
import tensorflow as tf
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(weights_path="./logs/ECG_best_weight/weights-best-epoch-50.weights.h5"):
    """Load the trained model with best weights"""
    try:
        model = create_model(input_shape=(300, 1))  # Giả sử look_back = 300
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

def visualize_predictions(y_true, y_pred, n_samples=5):
    """Visualize predicted vs actual ECG signals"""
    try:
        plt.figure(figsize=(15, 10))
        for i in range(min(n_samples, len(y_true))):
            plt.subplot(min(n_samples, len(y_true)), 1, i+1)
            plt.plot(y_true[i:i+300], label='Actual Signal', color='blue')
            plt.plot(y_pred[i:i+300], label='Predicted Signal', color='red', linestyle='--')
            plt.title(f'Sample {i+1}: Actual vs Predicted ECG Signal')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude')
            plt.legend()
        plt.tight_layout()
        plt.savefig('output_figure.png')  # Lưu hình ảnh
        plt.show()
        logger.info("Predictions visualized and saved as output_figure.png")
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        raise

def main():
    # Parameters
    data_path = "./data/processed/custom_training_dataset.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    weights_path = "./logs/ECG_best_weight/weights-best-epoch-50.weights.h5"  # Đường dẫn đến weights tốt nhất

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
        visualize_predictions(y_test[:n_samples], y_pred[:n_samples], n_samples=5)

        # Save evaluation metrics to file (optional)
        with open('evaluation_metrics.txt', 'w') as f:
            f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")
        logger.info("Evaluation metrics saved to evaluation_metrics.txt")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()