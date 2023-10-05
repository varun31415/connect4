import numpy as np

def aiplay(grid):
    # make ai play
    return grid

grid = np.zeros((6, 7))

end = False

while not end: 
    print(grid)
    col = int(input("Enter a column (left to right): "))

    for i in range(6):
        if grid[5-i][col-1] == 0:
            grid[5-i][col-1] = 1
            break