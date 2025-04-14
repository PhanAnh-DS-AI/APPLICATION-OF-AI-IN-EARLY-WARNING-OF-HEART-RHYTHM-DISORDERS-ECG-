# train.py
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model import create_model
from New_model import create_optimized_model
from preprocess_data import ECGDataProcessor  # Sửa tên file từ preprocess.py thành preprocess_data.py
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom callback để lưu trọng số tốt nhất mỗi 10 epoch
class SaveBestEveryNEpoch(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, period=10, **kwargs):
        super().__init__(filepath, monitor=monitor, save_best_only=save_best_only, **kwargs)
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:  # Lưu sau mỗi `self.period` epoch
            super().on_epoch_end(epoch, logs)

# Callback để đo thời gian huấn luyện
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.start = time.time()

    def on_train_end(self, logs={}):
        self.end = time.time()
        self.total_time = self.end - self.start
        logger.info(f"Total training time: {self.total_time:.2f} seconds")

def train_model():
    # Parameters 
    data_path = "./data/processed/data_102_filtered_100k.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    batch_size = 64
    epochs = 50

    # Initialize data processor
    processor = ECGDataProcessor(data_path, look_back, train_split, val_split)
    logger.info("Loading and processing ECG data...")
    X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

    # Create model
    logger.info("Creating model...")
    model = create_model(input_shape=(look_back, 1))

    # Đảm bảo thư mục logs và weights tồn tại
    output_dir = "./logs/ECG_best_weight"
    os.makedirs(output_dir, exist_ok=True)
    log_dir = "./logs/tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Tìm file weight gần nhất
    weight_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".h5")])
    if weight_files:
        latest_weight = os.path.join(output_dir, weight_files[-1])  # Chọn file gần nhất
        model.load_weights(latest_weight)  # Load trọng số
        logger.info(f"Loaded weights from {latest_weight}")
    else:
        logger.info("No previous weights found, training from scratch.")

    # Callback để lưu trọng số tốt nhất mỗi 10 epoch
    checkpoint_callback = SaveBestEveryNEpoch(
        filepath=os.path.join(output_dir, 'weights-best-epoch-{epoch:02d}.weights.h5'),
        monitor='val_mae',
        save_weights_only=True,
        save_best_only=True,
        period=10
    )

    # Callback TensorBoard
    tb_callback = TensorBoard(log_dir=log_dir)

    # Callback đo thời gian
    time_callback = TimeHistory()

    initial_epoch = int(weight_files[-1].split("-epoch-")[1].split(".")[0]) if weight_files else 0
    remaining_epochs = max(epochs - initial_epoch, 0)  # Đảm bảo không bị giá trị âm

    if remaining_epochs > 0:
        logger.info(f"Resuming training from epoch {initial_epoch + 1} to {epochs}")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  
            batch_size=batch_size,
            epochs=epochs,  # Tổng số epochs cần train
            initial_epoch=initial_epoch,  # Bắt đầu từ epoch đã train trước
            verbose=1,
            callbacks=[checkpoint_callback, tb_callback, time_callback]
        )
    else:
        logger.info("Training already completed, no remaining epochs to train.")
        
    # Đánh giá trên test set
    logger.info("Evaluating model on test set...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test MSE: {test_loss[0]:.4f}")
    logger.info(f"Test MAE: {test_loss[1]:.4f}")
    logger.info(f"Test RMSE: {test_loss[2]:.4f}")
    logger.info(f"Test AUC: {test_loss[3]:.4f}")


    return model, history

if __name__ == "__main__":
    model, history = train_model()
    
    
    