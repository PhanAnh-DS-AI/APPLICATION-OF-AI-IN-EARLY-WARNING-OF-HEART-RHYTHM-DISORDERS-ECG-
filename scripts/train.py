# train.py
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model import create_model
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
    data_path = "./data/processed/custom_training_dataset.csv"
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

    # Callback để lưu trọng số tốt nhất mỗi 10 epoch
    checkpoint_callback = SaveBestEveryNEpoch(
        filepath=os.path.join(output_dir, 'weights-best-epoch-{epoch:02d}.weights.h5'),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=10
    )

    # Callback TensorBoard
    tb_callback = TensorBoard(log_dir=log_dir)

    # Callback đo thời gian
    time_callback = TimeHistory()

    # Huấn luyện mô hình
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_callback, tb_callback, time_callback]
    )

    # Đánh giá trên test set
    logger.info("Evaluating model on test set...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test loss (MSE): {test_loss[0]}")

    return model, history

if __name__ == "__main__":
    model, history = train_model()