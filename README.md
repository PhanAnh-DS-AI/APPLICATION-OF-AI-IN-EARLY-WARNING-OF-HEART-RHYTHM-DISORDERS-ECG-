# ECG Signal Reconstruction with CNN-BiLSTM

This project implements a CNN-BiLSTM model for reconstructing ECG signals, focusing on improving signal reconstruction accuracy by leveraging a custom dataset with augmented abnormal samples. The model is trained on two datasets: the original MIT-BIH Arrhythmia dataset and a custom dataset with enhanced abnormal signals. The performance is evaluated using metrics such as MSE, MAE, RMSE, and R², and the training process is visualized using TensorBoard.

## Project Overview
The goal of this project is to reconstruct ECG signals using a hybrid CNN-BiLSTM model. The model consists of:
- **Residual CNN blocks**: Extract spatial features from raw ECG signals.
- **Bi-LSTM layers**: Capture temporal dependencies in the signal.
- **Dense layer**: Reconstruct the final ECG signal.

Two datasets are used for training:
- **Original Dataset**: Raw ECG signals from the MIT-BIH Arrhythmia database.
- **Custom Dataset**: Augmented dataset with additional abnormal samples to improve model generalization.

The model is evaluated on a test set (ECG-102) and an additional record (Record 214) to assess its generalization ability.

## Model Architecture
The architecture of the 1D CNN-BiLSTM model is illustrated below:
![Model Architecture](notebook_test/img/architecture.png)

- **Input**: Raw ECG signal (1D array).
- **CNN Blocks**: Residual convolutional layers to extract spatial features.
- **Bi-LSTM**: Bidirectional LSTM layers to model temporal dependencies.
- **Dense Layer**: Fully connected layer for signal reconstruction.

## Results
### Performance Metrics
The model was trained on both datasets, and the performance was evaluated on a test set. The results are summarized below:

| Metric | Original Dataset | Custom Dataset |
|--------|------------------|----------------|
| MSE    | 0.0007           | 0.0002         |
| MAE    | 0.0186           | 0.0108         |
| RMSE   | 0.0256           | 0.0140         |
| R²     | 0.9839           | 0.9952         |

**Additional Evaluation on Record 214** (30,000 data points from MIT-BIH Arrhythmia):

| Metric | Original Dataset | Custom Dataset |
|--------|------------------|----------------|
| MSE    | 0.0005           | 0.0002         |
| MAE    | 0.0146           | 0.0086         |
| RMSE   | 0.0220           | 0.0146         |
| R²     | 0.9568           | 0.9810         |

### Training Process (TensorBoard Visualizations)
The training process was monitored using TensorBoard, showing the loss, RMSE, and MAE over 50 epochs for both datasets.

#### Loss
![Loss (Original Dataset)](notebook_test/img/epoch_loss_original.png)
![Loss (Custom Dataset)](notebook_test/img/epoch_loss_custom.png)

#### RMSE
![RMSE (Original Dataset)](notebook_test/img/epoch_rmse_original.png)
![RMSE (Custom Dataset)](notebook_test/img/epoch_rmse_custom.png)

#### MAE
![MAE (Original Dataset)](notebook_test/img/epoch_mae_original.png)
![MAE (Custom Dataset)](notebook_test/img/epoch_mae_custom.png)

### Key Findings
- The model trained on the custom dataset outperforms the model trained on the original dataset across all metrics, with lower errors (MSE, MAE, RMSE) and a higher R² score.
- The custom dataset, with augmented abnormal samples, enables the model to learn more diverse and complex features, leading to better generalization (as seen in Record 214).
- TensorBoard visualizations show faster convergence and better stability on the custom dataset, especially on the validation set.

## How to Run
### Prerequisites
- Python 3.8+
- Required libraries: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `tensorboard`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset
- Place the original and custom ECG datasets in the `data/` directory:
  - `data/original_ecg.csv`: Original MIT-BIH Arrhythmia dataset.
  - `data/custom_ecg.csv`: Custom dataset with augmented abnormal samples.

### Training
1. Train the model on both datasets:
   ```bash
   python train_model.py --dataset original
   python train_model.py --dataset custom
   ```
2. Monitor the training process using TensorBoard:
   ```bash
   tensorboard --logdir logs/
   ```

### Evaluation
- The trained models will be saved in the `models/` directory.
- Evaluate the models on the test set and Record 214:
  ```bash
  python evaluate_model.py --model models/original_model.h5 --test_data data/test_ecg.csv
  python evaluate_model.py --model models/custom_model.h5 --test_data data/record_214.csv
  ```

## Contact
For any questions or contributions, feel free to reach out:
- Email: pnhatanh71@gmail.com
- GitHub Issues: [[Create an issue](https://github.com/your-username/your-repo/issues)](https://github.com/PhanAnh-DS-AI/APPLICATION-OF-AI-IN-EARLY-WARNING-OF-HEART-RHYTHM-DISORDERS-ECG-/issues)
