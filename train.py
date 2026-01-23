import torch
import torch.nn as nn
import torch.optim as optim

from Satellite_Origami.dataset_loader import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_loader(
    path=r"C:\Users\Sastra\OneDrive\文档\GitHub\Satellite_Origami\data\processed_images",
    batch_size=32,
    shuffle=True
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.LazyLinear(2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    print(f"\nEpoch {epoch+1} started")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

print("\nTraining finished!")
