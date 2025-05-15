from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_grid_search import model_grid_optimize
from preprocess_data import ECGDataProcessor
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải và xử lý dữ liệu
processor = ECGDataProcessor("./data/processed/custom_training_dataset.csv", look_back=300, train_split=0.7, val_split=0.15, device=device)
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

# Kết hợp tập train + val
X_train_full = np.concatenate((X_train.cpu().numpy(), X_val.cpu().numpy()))
y_train_full = np.concatenate((y_train.cpu().numpy(), y_val.cpu().numpy()))

# Chuyển dữ liệu sang tensor PyTorch
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)

# Tạo class Regressor dùng cho GridSearchCV
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.0001, l2_reg=0.001, epochs=50, batch_size=32, optimizer='adam'):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.model, _, criterion, _ = model_grid_optimize(
            input_shape=(300, 1),
            learning_rate=self.learning_rate,
            l2_reg=self.l2_reg
        )
        self.model.to(self.device)

        # Chọn optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        else:
            raise ValueError("Unsupported optimizer type")

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in tqdm(range(self.epochs), desc=f"Train {self.optimizer}, lr={self.learning_rate}"):
            total_loss = 0
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs.permute(0, 2, 1))
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Mỗi 10 epoch vẫn in loss như cũ
            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(loader):.4f}")
        return self

    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            preds = self.model(X.permute(0, 2, 1))
        return preds.cpu().numpy().flatten()

# Tạo lưới tham số
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [50, 70, 90],
    'learning_rate': [0.0001, 0.001, 0.01],
    'l2_reg': [0.0005, 0.001],
    'optimizer': ['adam', 'rmsprop', 'sgd'],
}

# Chạy Grid Search
grid = GridSearchCV(estimator=PyTorchRegressor(), param_grid=param_grid, cv=3,
                    scoring='neg_mean_squared_error', verbose=1)
grid_result = grid.fit(X_train_full_tensor, y_train_full_tensor)

# In kết quả tốt nhất
print(f"\n✅ Best parameters: {grid_result.best_params_}")
print(f"✅ Best score (MSE): {-grid_result.best_score_:.6f}")

# Lưu toàn bộ kết quả
results_df = pd.DataFrame(grid_result.cv_results_)
results_df.to_csv("grid_search_results.csv", index=False)
print("📁 Saved full grid search results to 'grid_search_results.csv'")

# Vẽ biểu đồ Learning Rate vs Score theo từng optimizer
plt.figure(figsize=(10, 6))
for opt in param_grid['optimizer']:
    subset = results_df[results_df['param_optimizer'] == opt]
    plt.plot(subset['param_learning_rate'], -subset['mean_test_score'], marker='o', label=opt)

plt.xscale('log')
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Negative MSE")
plt.title("Grid Search Results by Optimizer and Learning Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grid_search_plot.png")
plt.show()
print("📊 Saved plot to 'grid_search_plot.png'")
