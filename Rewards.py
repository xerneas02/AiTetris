import numpy as np
import hnswlib
from AccessMemory import *  # Utiliser les fonctions que tu as déjà pour accéder à la mémoire du jeu

# Fonction principale de récompense avec KNN pour l'exploration
def get_game_reward(pyboy, grid): 
    reward = get_score(pyboy)
    grid_reward = get_grid_reward(grid)*0.1
    return reward + grid_reward

def get_grid_reward(grid):
    # Add rewards for line clear and penalties for stacking high or leaving gaps
    # Penalize creating gaps
    penalty = 0
    for row in grid:
        if 0 in row and row.count(1) > 0:
            penalty -= 1  # Penalize rows with gaps

    return penalty    

def is_done(pyboy):
    return pyboy.memory[GAME_STATE] == 13