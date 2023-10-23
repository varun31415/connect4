import numpy as np

def aiplay(grid):
    # make ai play
    return grid

def checkforwin(grid, color, row_length = 4):
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


grid = np.zeros((6, 7))

end = False6

def playTurn(grid, i, color):
    for i in range(6):
        if grid[5-i][col-1] == 0:
            grid[5-i][col-1] = color
            break
    return grid

def evaluateGrid(grid, color):
    if checkforwins(grid,color) >= 1:
        return 0
    if checkforwins(grid,-color) >= 1:
        return 1
    value = 0
    value =+ 3 * checkforwins(grid, color, row_length=3) - checkforwins(grid, -color, row_length=3)
    value =+ 2 * checkforwins(grid, color, row_length=2) - checkforwins(grid, -color, row_length=2)
    value =+ checkforwins(grid, color, row_length=1) - checkforwins(grid, -color, row_length=1)
    value = 1 / value
    return value

while not end: 
    print(grid)
    wins = checkforwin(grid, 1)
    print(str(wins) + "wins \n")
    col = int(input("Enter a column (left to right): "))

    grid = playTurn(grid, i, color)
