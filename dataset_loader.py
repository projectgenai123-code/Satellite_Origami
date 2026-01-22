from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loader(path, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(
        root=path,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
