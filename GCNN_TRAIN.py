import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# =========================
# CONFIG
# =========================
IMAGE_SIZE = 64
CHANNELS = 1
Z_DIM = 100
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.0002

DATA_DIR = r"C:\Users\Sastra\OneDrive\文档\GitHub\Satellite_Origami\data\processed_images"
SAVE_DIR = "generated_patterns"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# DATA LOADER (NO LABELS NEEDED)
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(
    root=os.path.dirname(DATA_DIR),
    transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# GENERATOR
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# DISCRIMINATOR
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)                 # (B, 1, H, W)
        return out.view(x.size(0), -1).mean(dim=1)


# =========================
# INIT MODELS
# =========================
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    for batch_idx, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # ======================
        # Train Discriminator
        # ======================
        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
        fake_imgs = G(noise)

        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        d_real_loss = criterion(D_real, real_labels)
        d_fake_loss = criterion(D_fake, fake_labels)
        d_loss = d_real_loss + d_fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ======================
        # Train Generator
        # ======================
        output = D(fake_imgs)
        g_loss = criterion(output, real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    # ======================
    # SAVE GENERATED IMAGES
    # ======================
    with torch.no_grad():
        sample_noise = torch.randn(25, Z_DIM, 1, 1).to(device)
        fake_samples = G(sample_noise)
        save_image(
            fake_samples,
            f"{SAVE_DIR}/epoch_{epoch+1}.png",
            nrow=5,
            normalize=True
        )

print("✅ GAN training finished")

# =========================
# SAVE MODELS
# =========================
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
print("💾 Models saved")
