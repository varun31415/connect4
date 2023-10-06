# input: a 6x7 array where if a space is empty, it is 0.5, it is red, it is 1, and if it is yellow, it is 0
# output: a 7x1 array where the space with the highest value gets a piece put into

# 42 => linear 256 => relu => linear 256 => relu 7x1

# convolutions

# 6x7 => Convolution then pool 2d => Convolution then pool 2d ... => Linear to 7x1

# 6x7 => convolution kernel_size=2 in_channels=1, out_channels=8, padding=1, stride=1

#UNDERSTAND BATCHNUM
# UNDERSTAND SOFTMAX
# FEATURE FUNCTION (how to convert it into 3 feature input) feature.py
# create a neural network to predict connect 4

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Network(torch.nn.module):
    def __init__(self):
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), padding="same")
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3),
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=5),
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), padding="same")
        self.pool3 = torch.nn.MaxPool2d(kernel_size=7)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear1 = torch.nn.Linear(6*7, 64)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear2 = torch.nn.Linear(64, 7)
        self.relu5 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu5(x)
        return x




    
