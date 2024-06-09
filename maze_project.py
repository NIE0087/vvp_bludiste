import pandas as pd
import numpy as np 
from pathlib import Path
import random
from collections import deque
import matplotlib.pyplot as plt


class Maze:
    
    def __init__(self):
       
        """Function mainly for storing informations after loading the matrices (sets default values & passes arguments after objects are created)"""
        
        self.mazes = []
        self.maze_index = 1

    def load_mazes(self) -> list:
      
        """Function for loading mazes with true/false values from a file""" 
      
        data_path = Path("./data/")

        while True:
            maze_file = data_path.joinpath(f"maze_{self.maze_index}.csv")
            if not maze_file.exists():
                break

            maze = pd.read_csv(maze_file, header=None).to_numpy(dtype=bool)
            self.mazes.append(maze)
        
            self.maze_index += 1

        return self.mazes, self.maze_index

    def construct_incidence_matrix(self) -> list:
      
        """Function for transforming bool matrices into int matrices, where false=0 and true=1"""
       
        incidence_matrices = []

        for maze in self.mazes:
            size = maze.shape[1]
            incidence_matrix = np.zeros((size, size), dtype=int)
            for row_idx, row in enumerate(maze):
                for col_idx, value in enumerate(row):
                    if value:  # if the cell is True (represents an edge)
                        incidence_matrix[row_idx, col_idx] = 1

            incidence_matrices.append(incidence_matrix)
        return incidence_matrices

    def is_valid_point(self, matrix: list, row: int, col: int, visited: bool) -> bool:
       
        """Function for checking validity of neighbours in BFS algorithm"""
       
        size = len(matrix)
        
        # Check if the point is within the bounds of the matrix
        if row < 0 or row >= size or col < 0 or col >= size:
            return False
        
        # Check if the point corresponds to a zero in the matrix and is not visited
        if matrix[row][col] != 0 or visited[row][col]:
            return False
        
        return True

    def bfs_shortest_path(self, matrix: list) -> list:
       
        """Function for finding shortest path in mazes and constructing the path"""
       
        size = len(matrix)
        start = (0, 0)
        end = (size - 1, size - 1)
        visited = [[False] * size for _ in range(size)]
        d_row = [-1, 0, 1, 0]
        d_col = [0, 1, 0, -1]
        
        queue = deque([(start, [start])])  # tuple(cell(tuple), path(list of tuples))
        visited[start[0]][start[1]] = True

        while queue:
            cell, path = queue.popleft()
            if cell == end:
                return path

            for i in range(4):
                new_row = cell[0] + d_row[i]  # x coordination update
                new_col = cell[1] + d_col[i]  # y coordination update
                
                if self.is_valid_point(matrix, new_row, new_col, visited):  # check validity of updated coordinations
                    visited[new_row][new_col] = True
                    new_path = path + [(new_row, new_col)]  # update the path
                    queue.append(((new_row, new_col), new_path))

        return []  # If no path is found

    def plot_maze_and_path(self, maze: list, path: list, index: int) -> None:
       
        """Function for rendering the mazes and paths in them"""
       
        plt.imshow(maze, cmap='binary')
        plt.title(label=f"Maze {index}", loc='center')
        
        path_rows, path_cols = zip(*path)  
        plt.plot(path_cols, path_rows, color='red', linewidth=2)
        plt.axis('off')  
        plt.show()

    def maze_generator(self, maze: list) -> list:
       
        """Function for generating maze (if it runs more times, it will generate maze with more barriers)"""
       
        size = len(maze)
        
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            
            if (x == 0 and y == 0) or (x == size - 1 and y == size - 1):
                maze[x][y] = 0
            else:
                maze[x][y] = 1
                last_wall_coordinate = (x, y)
            
            if self.bfs_shortest_path(maze) == []:
                break
        
        return maze, last_wall_coordinate

    def maze_generator_helper(self, maze: list) -> list:
       
        """Function which deletes last barrier added in function Maze_generator""" 
        
        for _ in range(5):
            maze, last_wall = self.maze_generator(maze)
            x, y = last_wall
            maze[x][y] = 0
        
        return maze

