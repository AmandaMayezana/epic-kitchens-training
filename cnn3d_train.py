import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# Simple 3D CNN
# ------------------------------
class CNN3D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.fc = nn.Linear(16 * 8 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------------
# Simulated training
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake data: 50 clips of 16 frames each, 3x32x32 videos
    X = torch.randn(50, 3, 16, 32, 32)
    y = torch.randint(0, 10, (50,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CNN3D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training 3D CNN model...")
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

    torch.save(model.state_dict(), "cnn3d_model.pt")
    print("✅ Model saved as cnn3d_model.pt")

if __name__ == "__main__":
    main()


print("✅ 3D CNN training complete! Model saved at models/cnn3d_model.pt")
