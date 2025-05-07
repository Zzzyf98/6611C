import pandas as pd
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 构造对比样本对
class TimeSeriesContrastiveDataset(Dataset):
    def __init__(self, X, y, window_size=2, noise_level=0.01):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.noise_level = noise_level
        self.pos_pairs = []
        self.neg_pairs = []

        for i in range(len(X) - window_size + 1):
            a = X[i]
            p = X[i + 1]
            if y[i] == y[i + 1]:
                self.pos_pairs.append((a, p))

        for i in range(len(X) - 1):
            a = X[i]
            n = X[i + 1]
            if y[i] != y[i + 1]:
                self.neg_pairs.append((a, n))
            else:
                n_noisy = n + np.random.normal(0, self.noise_level, size=n.shape)
                self.neg_pairs.append((a, n_noisy))

    def __len__(self):
        return len(self.pos_pairs) + len(self.neg_pairs)

    def __getitem__(self, idx):
        if idx < len(self.pos_pairs):
            return torch.tensor(self.pos_pairs[idx][0], dtype=torch.float32), \
                torch.tensor(self.pos_pairs[idx][1], dtype=torch.float32), \
                torch.tensor(1, dtype=torch.long)
        else:
            return torch.tensor(self.neg_pairs[idx - len(self.pos_pairs)][0], dtype=torch.float32), \
                torch.tensor(self.neg_pairs[idx - len(self.pos_pairs)][1], dtype=torch.float32), \
                torch.tensor(0, dtype=torch.long)





# 1. 数据预处理
class LandslideDataset(Dataset):
    def __init__(self, features, labels, seq_length=10):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.labels) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def apply_wavelet_transform_cwt(data, wavelet='morl', scales=np.arange(1, 5)):
    transformed_data = []
    for col in data.columns:
        coefficients, _ = pywt.cwt(data[col], scales, wavelet)
        transformed_col = coefficients.T
        transformed_data.append(transformed_col)
    return np.hstack(transformed_data)

# 2. 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# 3. Focal Loss 定义
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs.squeeze(), targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        return F_loss

# 4. 训练与评估
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses, val_losses = [], []
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        val_losses.append(total_val_loss / len(val_loader))

        val_f1 = f1_score(all_labels, all_preds)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")  # 保存最佳模型
        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val F1: {val_f1:.4f}")

    model.load_state_dict(torch.load("best_model.pth"))
    return model, train_losses, val_losses

# 5. 主程序
def main():
    df = pd.read_csv("sensor_data_7_days.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    features = df[["roll", "pitch", "velocity"]].values
    labels = df["label"].values

    print("原始特征数据形状:", features.shape)

    transformed_features = apply_wavelet_transform_cwt(
        pd.DataFrame(features), wavelet='morl', scales=np.arange(1, 5)
    )
    print("小波变换后特征数据形状:", transformed_features.shape)

    scaler = StandardScaler()
    transformed_features = scaler.fit_transform(transformed_features)
    print("标准化后特征数据形状:", transformed_features.shape)

    X_train, X_val, y_train, y_val = train_test_split(transformed_features, labels, test_size=0.2, shuffle=False)

    TimeSeriesContrastiveDataset(X_train, y_train)

    train_dataset = LandslideDataset(X_train, y_train, seq_length=10)
    val_dataset = LandslideDataset(X_val, y_val, seq_length=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_preds)
    print(f"F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

if __name__ == "__main__":
    main()