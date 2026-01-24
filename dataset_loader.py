from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = 0  # dummy label
        image = self.transform(image)
        return image, label


def get_loader(path, batch_size=32, shuffle=True):
    dataset = UnlabeledImageDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
