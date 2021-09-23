import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, trasforms=transforms))
