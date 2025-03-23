import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Add, Bidirectional, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def create_optimized_model(input_shape=(300, 1)):
    inputs = Input(shape=input_shape)
    
    # Block 1: Convolutional block 
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)   # shape: (n,150, 64)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    # Block 2: Residual block CNN
    block_input = x  # shape: (150, 64)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(block_input)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(y)
    y = MaxPooling1D(pool_size=2)(y)   # shape: (75, 32)
    y = Dropout(0.2)(y)
    y = BatchNormalization()(y)
    
    # Projection cho block_input
    residual = MaxPooling1D(pool_size=2)(block_input)  # shape: (75, 64)
    residual = Conv1D(32, kernel_size=1, padding='same')(residual)  # shape: (75, 32)
    
    x = Add()([y, residual])  # kết quả shape: (75, 32)
    
    # Block 3: Residual block CNN 
    block_input = x  # shape: (75, 32)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(block_input)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(y)
    y = MaxPooling1D(pool_size=2)(y)   # shape: (37, 32) nếu pool_size=2 (lưu ý: 75//2 = 37)
    y = Dropout(0.2)(y)
    y = BatchNormalization()(y)
    
    # Projection cho block_input của 
    residual = MaxPooling1D(pool_size=2)(block_input)  # shape: (37, 32)
    # Ở đây số kênh đã bằng nhau nên không cần dùng Conv1D (hoặc có thể dùng nếu muốn)
    x = Add()([y, residual])  # kết quả shape: (37, 32)
    
    # Lớp Bi-LSTM với recurrent dropout
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, 
                                    kernel_regularizer=regularizers.l2(0.001),
                                    recurrent_dropout=0.2))(x)
    
    # Thêm Attention: sử dụng GlobalAveragePooling1D làm cách đơn giản để tổng hợp
    attn_out = GlobalAveragePooling1D()(lstm_out)
    
    # Lớp fully connected
    fc = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))(attn_out)
    fc = Dropout(0.3)(fc)
    outputs = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001),
                metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                        tf.keras.metrics.AUC(name='auc')])

    model.summary()
    return model

# Ví dụ chạy mô hình:
model = create_optimized_model()
