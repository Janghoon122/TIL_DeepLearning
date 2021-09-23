import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from Net import Net

'''데이터 로더'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform))

'''학습 시작'''
model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

model.train(train_loader, optimizer, criterion)