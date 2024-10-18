from AccessMemory import *
from Rewards import *

from multiprocessing import Pool

def minimax_non_threaded(grid, x, y, rot, n):
    if n == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        return {0: [mean_y]}
    
    next_states = get_next_states(grid, x, y, rot)

    if len(next_states) == 0:
        return {0: [-999999]}

    # Exécution normale sans threads pour les autres profondeurs
    evaluer = {i: list(minimax_non_threaded(next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1).values()) for i in next_states.keys()}

    # Si la clé 2 n'existe pas dans evaluer
    if 2 not in evaluer.keys():
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        evaluer[2] = [[mean_y + n * 0.01 + calculate_complete_lines(grid) * 10]]

    # Retourne la valeur maximale pour chaque état
    return {i: max(evaluer[i]) for i in evaluer.keys()}

# Version avec threads pour la première profondeur
def minimax(grid, x, y, rot, n):
    if n == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        return {0: [mean_y]}
    
    next_states = get_next_states(grid, x, y, rot)

    if len(next_states) == 0:
        return {0: [-999999]}

    # Paralléliser uniquement la première profondeur avec multiprocessing
    with Pool() as pool:
        results = pool.starmap(minimax_non_threaded, [(next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1) for i in next_states.keys()])

    evaluer = {i: list(result.values()) for i, result in zip(next_states.keys(), results)}

    # Si la clé 2 n'existe pas dans evaluer
    if 2 not in evaluer.keys():
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        evaluer[2] = [[mean_y + n * 0.01 + calculate_complete_lines(grid) * 10]]

    # Retourne la valeur maximale pour chaque état
    return {i: max(evaluer[i]) for i in evaluer.keys()}

#def minimax_from_states(state, n):
#    return minimax_non_threaded(state[3], state[1], state[2], state[0], n - 1)
