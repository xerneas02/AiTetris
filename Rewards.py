import numpy as np
import hnswlib
from AccessMemory import *  # Utiliser les fonctions que tu as déjà pour accéder à la mémoire du jeu

last_score = 0
# Fonction principale de récompense avec KNN pour l'exploration
def get_game_reward(pyboy): 
    global last_score 
    reward = get_score(pyboy) - last_score
    last_score = get_score(pyboy)
    lost = pyboy.memory[GAME_STATE] == 13
    #if lost:
    #    reward -= 50
    return reward, lost

def reset_reward():
    global last_score
    last_score = 0
    