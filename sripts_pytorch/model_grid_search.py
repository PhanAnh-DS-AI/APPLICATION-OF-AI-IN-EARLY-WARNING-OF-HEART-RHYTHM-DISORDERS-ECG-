import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Custom SMAPE metric
class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, y_true, y_pred):
        epsilon = 1e-8
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred) + epsilon) / 2.0
        return 100 * torch.mean(numerator / denominator)

# Main model building function for grid search
class ECGModel(nn.Module):
    def __init__(self, input_shape=(300, 1), l2_reg=0.001):
        super(ECGModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64)

        # Residual Block 2
        self.res_conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=10, padding='same')
        self.res_conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(32)
        self.residualpool1 = nn.MaxPool1d(kernel_size=2)
        self.residual1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, padding='same')

        # Residual Block 3
        self.res_conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, padding='same')
        self.res_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(32)
        self.residualpool2 = nn.MaxPool1d(kernel_size=2)
        self.residual2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, padding='same')

        # Bi-LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True, dropout=0.2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc1 = nn.Linear(128, 100)  # 64 * 2 for bidirectional LSTM
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.bn1(x)

        # Residual Block 2
        residual = self.residualpool1(x)
        residual = self.residual1(residual)
        y = F.relu(self.res_conv1(x))
        y = F.relu(self.res_conv2(y))
        y = self.pool2(y)
        y = self.dropout2(y)
        y = self.bn2(y)
        x = y + residual

        # Residual Block 3
        residual = self.residualpool2(x)
        residual = self.residual2(residual)
        y = F.relu(self.res_conv3(x))
        y = F.relu(self.res_conv4(y))
        y = self.pool3(y)
        y = self.dropout3(y)
        y = self.bn3(y)
        x = y + residual

        # LSTM layer
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_len, input_size) format
        x, _ = self.lstm(x)  # LSTM expects input with shape (batch_size, seq_len, input_size)
        
        # Global Average Pooling
        x = x.permute(0, 2, 1)  # chuyển về (batch_size, features, seq_len)
        x = self.global_pool(x)  # sẽ ra (batch_size, features, 1)
        x = x.view(x.size(0), -1)  # → (batch_size, features)

        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

# Model initialization
def model_grid_optimize(input_shape=(300, 1), learning_rate=0.0001, l2_reg=0.001):
    model = ECGModel(input_shape=input_shape, l2_reg=l2_reg)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    # Loss function
    criterion = nn.MSELoss()

    # SMAPE metric
    smape = SMAPE()

    return model, optimizer, criterion, smape


if __name__ == "__main__":
    # Example usage
    model, optimizer, criterion, smape = model_grid_optimize()
    print(model)