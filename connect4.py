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
import pandas as pd
import numpy as np
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

model = Network()
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(grid, loss_fn, optimizer):
    model.train()
    train_loss = 0
    correct = 0

    X = grid

    optimizer.zero_grad()

    pred = model(X)
    ind = np.argpartition(pred, -4)[-4:]
    new_x = playTurn(X, ind[np.argsort(pred[ind])][0], -1)
    
    loss = evaluateGrid(new_x, -1)

    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    if batch % 100 == 0:
        print(f"loss: {loss:>7f} [{(batch + 1):>5d}/{len(dataloader):>6d}]")
        
    print(f"\n\n\nAverage loss: {train_loss/len(dataloader)}\nAccuracy: {100*correct/len(dataloader)}")

    return new_x


def aiplay(grid):
    # make ai play
    return grid

def checkforwins(grid, color, row_length = 4):
    wins = 0
    for i in range(6):
        for j in range(6):
            if grid[i,j] == color:
                
                # check for diagnols
                for k in range(row_length-1):
                    try: 
                        if grid[i+1+k,j+1+k] != color:
                            break
                        if k == 2:
                            wins = wins + 1
                    except: 
                        break
                #check for columns
                for k in range(row_length-1):
                    try: 
                        if grid[i,j+1+k] != color:
                            break
                        if k == 2:
                            wins = wins + 1
                    except: 
                        break
                #check for rows
                for k in range(row_length-1):
                    try: 
                        if grid[i+1+k,j] != color:
                            break
                        if k == 2:
                            wins = wins + 1
                    except: 
                        break
    return wins



end = False

def playTurn(ingrid, i, color):
    for i in range(6):
        if ingrid[5-i][col-1] == 0:
            ingrid[5-i][col-1] = color
            break
    return ingrid

def evaluateGrid(grid, color):
    if checkforwins(grid,color) >= 1:
        return 0
    if checkforwins(grid,-color) >= 1:
        return 2
    value = 0
    value =+ 3 * checkforwins(grid, color, row_length=3) - checkforwins(grid, -color, row_length=3)
    value =+ 2 * checkforwins(grid, color, row_length=2) - checkforwins(grid, -color, row_length=2)
    value =+ checkforwins(grid, color, row_length=1) - checkforwins(grid, -color, row_length=1)
    return value

grid = np.zeros((6, 7))

while not end: 
    print(grid)
    wins = checkforwins(grid, 1)
    print(str(wins) + "wins \n")
    col = int(input("Enter a column (left to right): "))

    grid = playTurn(grid, col, 1)

    grid = train(grid, loss_fn, optimizer)
    #lowest_loss = 100
    #finalpred = 1
    #for pred in range(7):
    #    pred = pred + 1
    #    loss = evaluateGrid(playTurn(grid, pred, -1), -1)
     #   print(loss, pred)
    #    if loss < lowest_loss:
      #      print(pred)
     #       finalpred = pred
     #       lowest_loss = loss
    #grid = playTurn(grid, finalpred, -1)
