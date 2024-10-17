import numpy as np
import hnswlib
from AccessMemory import *  # Utiliser les fonctions que tu as déjà pour accéder à la mémoire du jeu

# Fonctions d'aide pour les heuristiques
def calculate_aggregate_height(grid):
    heights = [0] * len(grid[0])
    for col in range(len(grid[0])):
        for row in range(len(grid)):
            if grid[row][col]:
                heights[col] = len(grid) - row
                break
    return sum(heights)

def calculate_holes(grid):
    holes = 0
    for col in range(len(grid[0])):
        block_found = False
        for row in range(len(grid)):
            if grid[row][col]:
                block_found = True
            elif block_found and not grid[row][col]:
                holes += 1
    return holes

def calculate_bumpiness(grid):
    heights = [0] * len(grid[0])
    for col in range(len(grid[0])):
        for row in range(len(grid)):
            if grid[row][col]:
                heights[col] = len(grid) - row
                break
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def calculate_complete_lines(grid):
    complete_lines = 0
    for row in grid:
        if all(row):
            complete_lines += 1
    return complete_lines

# Fonction de récompense améliorée
def get_grid_reward(grid):
    grid = [[1 if grid[i][j] == 2 else grid[i][j] for j in range(len(grid[i]))] for i in range(len(grid))]
    
    aggregate_height = calculate_aggregate_height(grid)
    holes = calculate_holes(grid)
    bumpiness = calculate_bumpiness(grid)
    complete_lines = calculate_complete_lines(grid)
    
    # Poids heuristiques (à ajuster)
    GA_ParamA = -0.510066
    GA_ParamB = 0.760666
    GA_ParamC = -0.35663
    GA_ParamD = 0.760666  
    
    # Calcul du score heuristique
    heuristic_score = (GA_ParamA * aggregate_height) + (GA_ParamB * holes) + (GA_ParamC * bumpiness) + (GA_ParamD * complete_lines)
    
    return heuristic_score

def get_game_reward(pyboy, grid, n_pieces): 
    reward = get_score(pyboy)*0 + n_pieces - 20
    heuristic_reward = get_grid_reward(grid)
    return reward + heuristic_reward

"""
def get_grid_reward(grid):
    # Add rewards for line clear and penalties for stacking high or leaving gaps
    # Penalize creating gaps
    penalty = 0
    for row in grid:
        if 0 in row and row.count(1) > 0:
            penalty -= row.count(0) # Penalize rows with gaps

    return penalty 
"""   

def is_done(pyboy):
    return pyboy.memory[GAME_STATE] == 13