import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader import get_loader


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Data Loader
# -----------------------------
train_loader = get_loader(
    path=r"C:\Users\Sastra\OneDrive\文档\GitHub\Satellite_Origami\data\processed_images_split\train",
    batch_size=32,
    shuffle=True
)


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.LazyLinear(2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)


# -----------------------------
# Training Setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# Training Loop
# -----------------------------
epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

print("\n✅ Training finished successfully!")
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved as trained_model.pth")

