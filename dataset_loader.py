import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_loader

# 1️⃣ Load your 2000 images (in batches)
train_loader = get_loader(
    path="C:\\Users\\Sarika Kannan\\satellite_origami\\Satellite_Origami\\data\\processed_images",
    batch_size=32,
    shuffle=True
)

# 2️⃣ Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(16 * 32 * 32, 2)  # 2 classes

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()

# 3️⃣ Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4️⃣ Training loop
for epoch in range(5):  # 5 epochs
    print(f"\nEpoch {epoch+1} started")

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images = [32, 1, 64, 64]
        # labels = [32]

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch+1}, "
                f"Batch {batch_idx}, "
                f"Loss: {loss.item():.4f}"
            )

print("\nTraining finished!")
