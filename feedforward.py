# imports
import torch
import torch.nn as nn
from torch.nn.modules import module
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784  # 28x28 input of conv img
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learing_rate = 0.001


# MNIST data
train_dataset = MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size)

# X, y = next(iter(train_loader))
# print(X[1].shape)
# print(y[1])

# examples = iter(train_loader)
# samples, lables = examples.next()
# print(samples.shape, lables.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0])
# plt.show()


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


model = NeuralNet(input_size, hidden_size, num_classes)
model = model.to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)


# # training loop
# n_total_steps = len(train_loader)
# losses = []
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, (images, lables) in enumerate(train_loader):
#         images = images.reshape(-1, 28*28).to(device)
#         lables = lables.to(device)

#         # forward
#         outputs = model(images)
#         loss = criterion(outputs, lables)

#         # backwards
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)

#         if(i+1) % 100 == 0:
#             print(
#                 f'=> epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f} ')

#     epoch_loss = running_loss / len(train_loader)
#     losses.append(epoch_loss)
#     print(f'Epoch {epoch+1} loss : {epoch_loss:.4f}')

# print(losses)
# plt.plot(losses)
# plt.show()


# # testing loop
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, lables in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         lables = lables.to(device)
#         outputs = model(images)

#         _, predictions = torch.max(outputs, 1)  # 1 is the lables
#         n_samples += lables.shape[0]  # 0 is samples per batch
#         n_correct += (predictions == lables).sum().item()

#     acc = 100 * n_correct / n_samples
#     print(f'accuracy = {acc}')


def train():
    n_total_steps = len(train_loader)
    train_losses = []
    val_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        test_acc = 0.0
        train_acc = 0.0

        # train loop
        for i, (images, lables) in enumerate(train_loader):
            images = images.reshape(images.shape[0], -1).to(device)
            lables = lables.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, lables)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if(i+1) % 100 == 0:
                print(
                    f'=> epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f} ')

        epoch_train_loss = train_loss / len(train_loader)

        # validation loop
        model.eval()
        for i, (images, lables) in enumerate(test_loader):
            images = images.reshape(images.shape[0], -1).to(device)
            lables = lables.to(device)

            outputs = model(images)
            loss = criterion(outputs, lables)

            val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(test_loader)

        with torch.no_grad():
            n_correct_test = 0
            n_samples_test = 0
            for images, lables in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                lables = lables.to(device)
                outputs = model(images)

                _, predictions = torch.max(outputs, 1)  # 1 is the lables
                n_samples_test += lables.shape[0]  # 0 is samples per batch
                n_correct_test += (predictions == lables).sum().item()

            test_acc = 100 * n_correct_test / n_samples_test
            #print(f'accuracy = {acc}')
            n_correct_train = 0
            n_samples_train = 0
            for images, lables in train_loader:
                images = images.reshape(-1, 28*28).to(device)
                lables = lables.to(device)
                outputs = model(images)

                _, predictions = torch.max(outputs, 1)  # 1 is the lables
                n_samples_train += lables.shape[0]  # 0 is samples per batch
                n_correct_train += (predictions == lables).sum().item()

            train_acc = 100 * n_correct_train / n_samples_train

        print(
            f'Epoch {epoch+1} => Train loss : {epoch_train_loss:.4f} => Val loss : {epoch_val_loss:.4f} => Test acc : {test_acc:.2f} % => Train acc : {train_acc:.2f} %')

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plt.plot(train_losses, 'r')
    plt.plot(val_losses, 'b')
    plt.show()
    plt.plot(train_accs, 'm')
    plt.plot(test_accs, 'y')
    plt.show()


train()

PATH = 'enitre_model.pth'
torch.save(model, PATH)
