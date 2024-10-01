import numpy as np
import hnswlib
from AccessMemory import *  # Utiliser les fonctions que tu as déjà pour accéder à la mémoire du jeu

# Fonction principale de récompense avec KNN pour l'exploration
def get_game_reward(pyboy): 
    reward = get_score(pyboy)
    return reward

def is_done(pyboy):
    return pyboy.memory[GAME_STATE] == 13