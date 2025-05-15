import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model_grid_search import model_grid_optimize
from preprocess_data import ECGDataProcessor
import logging
from tqdm import tqdm  # <--- THÊM tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Callback để đo thời gian huấn luyện
class TimeHistory:
    def on_train_begin(self):
        self.start = time.time()

    def on_train_end(self):
        self.end = time.time()
        self.total_time = self.end - self.start
        logger.info(f"Total training time: {self.total_time:.2f} seconds")

def train_model():
    # Parameters
    data_path = "./data/processed/custom_training_dataset.csv"
    look_back = 300
    train_split = 0.7
    val_split = 0.15
    batch_size = 32
    epochs = 70
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard writer
    writer = SummaryWriter(log_dir="./logs/tensorboard")

    # Data loading
    processor = ECGDataProcessor(data_path, look_back, train_split, val_split, device)
    logger.info("Loading and processing ECG data...")
    X_train, y_train, X_val, y_val, X_test, y_test = processor.process_and_get_data()

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model + optimizer + criterion + smape
    logger.info("Creating model...")
    model, optimizer, criterion, smape = model_grid_optimize(input_shape=(look_back, 1))
    model.to(device)

    # Output weight directory
    output_dir = "./logs/ECG_best_weight"
    os.makedirs(output_dir, exist_ok=True)

    # Resume training if possible
    weight_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".pt")])
    if weight_files:
        latest_weight = os.path.join(output_dir, weight_files[-1])
        model.load_state_dict(torch.load(latest_weight))
        logger.info(f"Loaded weights from {latest_weight}")
    else:
        logger.info("No previous weights found, training from scratch.")

    # Training loop
    time_callback = TimeHistory()
    time_callback.on_train_begin()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Hiển thị tiến trình training
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", colour="cyan" ,leave=True)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.transpose(1, 2))  # Transpose for Conv1d

            # Sửa lỗi shape: [B,1] vs [B]
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_smape_total = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.transpose(1, 2))
                val_loss += criterion(outputs.squeeze(), labels).item()
                val_smape_total += smape(labels, outputs.squeeze()).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_smape = val_smape_total / len(val_loader)

        # Logging
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val SMAPE: {avg_val_smape:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        writer.add_scalar("SMAPE/Val", avg_val_smape, epoch + 1)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(output_dir, f"weights-best-epoch-{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model at epoch {epoch+1} to {save_path}")

    time_callback.on_train_end()

    # Final test evaluation
    logger.info("Evaluating model on test set...")
    model.eval()
    test_loss = 0.0
    test_smape_total = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.transpose(1, 2))
            test_loss += criterion(outputs.squeeze(), labels).item()
            test_smape_total += smape(labels, outputs.squeeze()).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_smape = test_smape_total / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Test SMAPE: {avg_test_smape:.4f}")

    writer.add_scalar("Loss/Test", avg_test_loss, epochs)
    writer.add_scalar("SMAPE/Test", avg_test_smape, epochs)

    writer.close()
    return model

if __name__ == "__main__":
    model = train_model()
    logger.info("Training completed.")
