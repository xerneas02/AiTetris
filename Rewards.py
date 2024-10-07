import numpy as np
import hnswlib
from AccessMemory import *  # Utiliser les fonctions que tu as déjà pour accéder à la mémoire du jeu

# Fonction principale de récompense avec KNN pour l'exploration
def get_game_reward(pyboy, grid): 
    reward = get_score(pyboy)
    grid_reward = get_grid_reward(grid)*0.1
    return reward + grid_reward

def get_grid_reward(grid):
    reward = 0
    for i in range(len(grid)):
        count = grid[i].count(1)
        if count > 0:
            reward += -(len(grid[0]) - count)*i
    return reward        

def is_done(pyboy):
    return pyboy.memory[GAME_STATE] == 13