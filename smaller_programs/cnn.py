import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size=128

test_dataloader = DataLoader(test_data, batch_size=batch_size)
train_dataloader = DataLoader(training_data, batch_size=batch_size)


for x, y in test_dataloader:
    print(f"Shape of X: {x.shape}")
    print(f"Shape of Y: {y.shape}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear = nn.Sequential(
            nn.Linear(24*7*7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()/128#error here
        if batch % 100 == 0:
            print(f"loss: {loss:>7f} [{(batch + 1):>5d}/{len(dataloader):>6d}]")

    print(f"correct: {correct}, dataloadersize: {len(dataloader)}")
    print(f"\n\n\nAverage loss: {train_loss/len(dataloader)}\nAccuracy: {100*correct/len(dataloader)}")


epochs = 100

for i in range(epochs):
    print(f"\n\nEpoch {i+1}-----------------------")
    train(train_dataloader, model, loss_fn, optimizer)

torch.save(model.state_dict(), "convolution_digit_rec.pth")
print("Saved to convolution_digit_rec.pth")