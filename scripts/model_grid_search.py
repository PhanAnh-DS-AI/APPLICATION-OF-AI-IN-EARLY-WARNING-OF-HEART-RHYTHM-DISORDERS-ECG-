import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Add, Bidirectional, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

# Custom SMAPE metric
class SMAPE(Metric):
    def __init__(self, name="smape", **kwargs):
        super(SMAPE, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = K.epsilon()
        numerator = K.abs(y_pred - y_true)
        denominator = (K.abs(y_true) + K.abs(y_pred) + epsilon) / 2.0
        value = numerator / denominator
        self.total.assign_add(K.sum(value))
        self.count.assign_add(K.cast(K.size(y_true), K.floatx()))

    def result(self):
        return 100 * self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# Main model building function for grid search
def model_grid_optimize(input_shape=(300, 1), learning_rate=0.0001, l2_reg=0.001):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Residual Block 2
    block_input = x
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(block_input)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Dropout(0.2)(y)
    y = BatchNormalization()(y)

    residual = MaxPooling1D(pool_size=2)(block_input)
    residual = Conv1D(32, kernel_size=1, padding='same')(residual)

    x = Add()([y, residual])

    # Residual Block 3
    block_input = x
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(block_input)
    y = Conv1D(32, kernel_size=10, activation='relu', padding='same')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Dropout(0.2)(y)
    y = BatchNormalization()(y)

    residual = MaxPooling1D(pool_size=2)(block_input)

    x = Add()([y, residual])

    # Bi-LSTM layer
    lstm_out = Bidirectional(LSTM(64, return_sequences=True,
                                   kernel_regularizer=regularizers.l2(l2_reg),
                                   recurrent_dropout=0.2))(x)

    # Global Average Pooling for Attention
    attn_out = GlobalAveragePooling1D()(lstm_out)

    # Fully connected
    fc = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(attn_out)
    fc = Dropout(0.3)(fc)
    outputs = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_reg))(fc)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(name='mae'),
                      tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                      tf.keras.metrics.AUC(name='auc'),
                      SMAPE()  # Custom SMAPE
                  ])
    
    model.summary()
    return model
