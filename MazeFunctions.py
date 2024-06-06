
import pandas as pd
import numpy as np 
from pathlib import Path
import random
from collections import deque
import matplotlib.pyplot as plt


def load_mazes()-> list|int:
    
    """function for loading mazes with true/false values from a file""" 
    
    data_path = Path("./data/")
    maze_index = 1
    mazes = []

    while True:
        maze_file = data_path.joinpath(f"maze_{maze_index}.csv")
        if not maze_file.exists():
            break

        maze = pd.read_csv(maze_file, header=None).to_numpy(dtype=bool)
        mazes.append(maze)
    
        maze_index += 1

    return mazes, maze_index


def construct_incidence_matrix(mazes: list) -> list: #it really isn't an incidence matrix, just couldn't think of any other name 
   
    """function for transforming bool matrices into int matrices, where false=0 and true=1"""
    
    incidence_matrices = []

    for _, maze in enumerate(mazes):
         n = maze.shape[1]
         incidence_matrix = np.zeros((n, n), dtype=int)
         for row_idx, row in enumerate(maze):
            for col_idx, value in enumerate(row):
                if value:  # if the cell is True (represents an edge)
                    incidence_matrix[row_idx, col_idx] = 1
                      

         incidence_matrices.append(incidence_matrix)
    return incidence_matrices




def is_valid_point(matrix: list, row: int, col: int, visited: bool) -> bool:
  
    """function for checking validity of neighbours in BFS algorithm"""
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Check if the point is within the bounds of the matrix
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return False
    
    # Check if the point corresponds to a zero in the matrix and is not visited
    if matrix[row][col] != 0 or visited[row][col]:
        return False
    
    return True


def bfs_shortest_path(matrix:list) -> list: 
  
    """function for finding shortest path in mazes and constructing the path"""
  
    n = len(matrix)
    start = (0,0)
    end=(n-1,n-1)
    visited = [[False] * n for _ in range(n)]
    dRow = [-1, 0, 1, 0]
    dCol = [0, 1, 0, -1]
    
    queue = deque([(start, [start])])  # tuple(cell(tuple), path(list of tuples))
    visited[start[0]][start[1]] = True

    while queue:
        cell, path = queue.popleft() #
        if cell == end:
            return path

        for i in range(4):

            new_row = cell[0] + dRow[i] #x coordination update
            new_col = cell[1] + dCol[i] #y coordination update
            
            if is_valid_point(matrix, new_row, new_col, visited): #check validity of updated coordinations
                
                visited[new_row][new_col] = True
               
                new_path = path + [(new_row, new_col)]  # update the path
                queue.append(((new_row, new_col), new_path)) 

    return [] # If no path is found



def plot_of_maze_and_path(maze: list, path: list, index: int) -> None:
  
    """function for rendering the mazes and paths in them"""
  
    plt.imshow(maze, cmap='binary')
    plt.title(label=f"Maze {index}",loc='center')
    
    path_rows, path_cols = zip(*path)  
    plt.plot(path_cols, path_rows, color='red', linewidth=2)
    plt.axis('off')  
   
    plt.show()



def Maze_generator(maze: list) -> list|tuple:
  
    """function for generating maze (if it runs more times, it will generate maze with more barriers)"""
 
    n = len(maze)
    
    while True:
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        
        if (x == 0 and y == 0) or (x == n - 1 and y == n - 1):
            maze[x][y] = 0
        else:
            maze[x][y] = 1
            last_wall_coordinate = (x, y)
        
        if bfs_shortest_path(maze)==[]:
            break
    
    return maze, last_wall_coordinate


def Maze_generator_helper(maze: list)-> list:
   
    """function which deletes last barrier added in function Maze_generator""" 
    for i in range(5):
        maze, last_wall = Maze_generator(maze)
        x, y = last_wall
        maze[x][y] = 0
    
    return maze

