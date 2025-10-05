import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RealVideoDataset(Dataset):
    def __init__(self, video_dir, label_dict, num_frames=16, transform=None):
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.label_dict = label_dict
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        frames = self._load_frames(path)  # shape: (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # convert to (C, T, H, W)
        label = self.label_dict[os.path.basename(path)]
        return frames, label

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(total)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = torch.zeros(3, 112, 112)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)  # shape: (T, C, H, W)

    def _sample_indices(self, total):
        if total < self.num_frames:
            return list(range(total)) + [total - 1] * (self.num_frames - total)
        step = total // self.num_frames
        return [i * step for i in range(self.num_frames)]


video_dir = "C:/EPIC-KITCHENS/EPIC-KITCHENS/P01/videos"
label_dict = {
    "P01_01.MP4": 0,
    "P01_02.MP4": 1,
    # Add more mappings
}

dataset = RealVideoDataset(video_dir, label_dict)
loader = DataLoader(dataset, batch_size=2, shuffle=True)


model = r3d_18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(3):
    total_loss = 0
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn3d_model.pt")

print("✅ 3D CNN training complete! Model saved at models/cnn3d_model.pt")
