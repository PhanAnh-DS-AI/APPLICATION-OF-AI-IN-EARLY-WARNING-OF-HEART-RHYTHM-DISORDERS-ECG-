# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense, Bidirectional, BatchNormalization, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def create_model(input_shape=(300, 1)):
    """Create the 1D CNN + Bi-LSTM model for ECG signal reconstruction"""
    model = Sequential()
    
    # Khối CNN đầu tiên với input_shape
    model.add(Conv1D(32, 5, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Các khối CNN tiếp theo (2 khối)
    for _ in range(2):
        model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
        model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

    # Lớp Bi-LSTM
    model.add(Bidirectional(LSTM(50, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Lớp đầu ra
    model.add(Dense(1, activation="linear", kernel_regularizer=regularizers.l2(0.001)))

    # Compile model
    model.compile(
        loss='mean_squared_error',  # RMSE
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()
    return model