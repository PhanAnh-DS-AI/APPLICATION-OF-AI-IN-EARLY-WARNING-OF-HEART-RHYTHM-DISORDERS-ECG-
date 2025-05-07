import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGDataProcessor:
    def __init__(self, data_path: str, look_back: int, train_split: float, val_split: float):
        self.data_path = data_path
        self.look_back = look_back
        self.train_split = train_split
        self.val_split = val_split
        self.data = None

    def low_pass_filter(self, signal, cutoff=40, fs=500, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered = filtfilt(b, a, signal.flatten())
        return filtered.reshape(-1, 1)

    def load_and_validate_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            if self.data.empty:
                raise ValueError("Empty dataset")

            self.data = self.data.rename(columns={self.data.columns[0]: "V2"})

            if self.data.isnull().any().any():
                logger.warning("Missing values detected. Filling with forward fill")
                self.data = self.data.fillna(method='ffill')

            self.data = self.data["V2"].to_numpy().reshape(-1, 1)

            # Step 1: Apply low-pass filter
            self.data = self.low_pass_filter(self.data)

            # Step 2: Normalize using Min-Max Scaler to [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.data = scaler.fit_transform(self.data)

            logger.info(f"Data loaded, filtered, and normalized successfully. Shape: {self.data.shape}")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_sequences(self, data):
        X, Y = [], []
        try:
            for i in range(len(data) - self.look_back):
                X.append(data[i:(i + self.look_back), 0])
                Y.append(data[i + self.look_back, 0])

            X, Y = np.array(X), np.array(Y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            return X, Y
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

    def split_and_prepare_dataset(self):
        try:
            total_length = len(self.data)
            if total_length < self.look_back * 3:
                raise ValueError("Dữ liệu quá ngắn cho look_back và split")

            train_size = int(total_length * self.train_split)
            val_size = int(total_length * self.val_split)

            train_data = self.data[:train_size]
            val_data = self.data[train_size:train_size + val_size]
            test_data = self.data[train_size + val_size:]

            X_train, y_train = self.create_sequences(train_data)
            X_val, y_val = self.create_sequences(val_data)
            X_test, y_test = self.create_sequences(test_data)

            logger.info(f"Training set shape: {X_train.shape}")
            logger.info(f"Validation set shape: {X_val.shape}")
            logger.info(f"Test set shape: {X_test.shape}")

            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise

    def process_and_get_data(self, data_path=None):
        if data_path:
            self.data_path = data_path
        self.load_and_validate_data()
        return self.split_and_prepare_dataset()


if __name__ == "__main__":
    data_path = "./data/processed/custom_training_dataset.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15

    processor = ECGDataProcessor(data_path, look_back=look_back, train_split=train_split, val_split=val_split)
    X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
