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

class Network(torch.nn.module):
    def __init__(self):
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), padding="same")
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), padding="same")
        self.relu2 = torch.nn.ReLU(inplace=True)