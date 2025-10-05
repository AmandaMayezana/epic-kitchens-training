import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms

class RealVideoDataset(Dataset):
    def __init__(self, video_dir, label_dict, num_frames=16, transform=None):
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.label_dict = label_dict  # e.g., {'video1.mp4': 0, 'video2.mp4': 1}
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self._load_frames(video_path)
        label = self.label_dict[os.path.basename(video_path)]
        return frames, label

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._sample_indices(total_frames)

        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = torch.zeros(3, 112, 112)  # fallback
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)  # shape: (num_frames, 3, 112, 112)

    def _sample_indices(self, total):
        if total < self.num_frames:
            return list(range(total)) + [total - 1] * (self.num_frames - total)
        step = total // self.num_frames
        return [i * step for i in range(self.num_frames)]


video_dir = "path/to/videos"
label_dict = {
    "P01_001.mp4": 0,
    "P01_002.mp4": 1,
    # Add all your video-label mappings
}

dataset = RealVideoDataset(video_dir, label_dict)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


# CNN for feature extraction (using pretrained ResNet)
cnn = models.resnet18(pretrained=True)
cnn.fc = nn.Identity()  # remove final layer
for param in cnn.parameters():
    param.requires_grad = False  # freeze CNN

# LSTM for temporal modeling
lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
classifier = nn.Linear(128, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(lstm.parameters()) + list(classifier.parameters()), lr=0.001)

# Training loop
for epoch in range(3):
    total_loss = 0
    for frames, labels in loader:
        optimizer.zero_grad()
        # Extract features frame by frame
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)  # flatten temporal dimension
        features = cnn(frames)               # shape: (B*T, 512)
        features = features.view(B, T, -1)   # reshape back to (B, T, 512)
        outputs, _ = lstm(features)
        preds = classifier(outputs[:, -1, :])
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save({
    "cnn": cnn.state_dict(),
    "lstm": lstm.state_dict(),
    "classifier": classifier.state_dict(),
}, "models/cnn_lstm_model.pt")

print("✅ CNN-LSTM training complete! Model saved at models/cnn_lstm_model.pt")
