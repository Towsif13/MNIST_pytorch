import torch
import torch.nn as nn
from torch.nn.modules import module
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

test_dataset = MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

test_loader = DataLoader(
    dataset=test_dataset, batch_size=100)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out


PATH = 'enitre_model.pth'
model = torch.load(PATH)
model.eval()

# #loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, lables in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # 1 is the lables
        n_samples += lables.shape[0]  # 0 is samples per batch
        n_correct += (predictions == lables).sum().item()

    acc = 100 * n_correct / n_samples
    print(f'accuracy = {acc} %')
