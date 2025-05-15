from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from model_grid_search import model_grid_optimize
from preprocess_data import ECGDataProcessor
import numpy as np

# Tải và xử lý dữ liệu
processor = ECGDataProcessor("./data/processed/custom_training_dataset.csv", look_back=300, train_split=0.7, val_split=0.15)
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

# Kết hợp train + val để huấn luyện
X_train_full = np.concatenate((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# Đóng gói mô hình với hàm xây dựng
def build_model(learning_rate=0.0001, l2_reg=0.001):
    return model_grid_optimize(input_shape=(300, 1), learning_rate=learning_rate, l2_reg=l2_reg)

model = KerasRegressor(build_fn=build_model, epochs=50, batch_size=64, verbose=0)

# Tạo lưới tham số
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [50, 70, 100],
    'learning_rate': [0.0001, 0.001,0.01],
    'l2_reg': [0.0005, 0.001],
    'optimizer': ['adam', 'rmsprop', 'sgd'],
}

# Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_result = grid.fit(X_train_full, y_train_full)

# In kết quả tốt nhất
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")
