import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# Simple CNN-LSTM architecture
# ------------------------------
class CNNLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_classes=10):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(b, t, -1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

# ------------------------------
# Simulated training
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake data: 50 sequences of 10 frames each, 3x64x64 images
    X = torch.randn(50, 10, 3, 64, 64)
    y = torch.randint(0, 10, (50,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training CNN-LSTM model...")
    for epoch in range(args.epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "cnn_lstm_model.pt")
    print("✅ Model saved as cnn_lstm_model.pt")

if __name__ == "__main__":
    main()
