from AccessMemory import *
from Rewards import *

from multiprocessing import Pool

def reorder_list(tab):
    order = [3, 0, 1, 2]
    return [x for x in order if x in tab]

def minimax_non_threaded(grid, x, y, rot, n, hmap_pos):
    if n == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        return {0: [mean_y]}
    
    next_states = get_next_states(grid, x, y, rot)

    for keys in list(next_states.keys()):
        if f"{next_states[keys][1]}{next_states[keys][2]}{next_states[keys][0]}" in hmap_pos.keys() and hmap_pos[f"{next_states[keys][1]}{next_states[keys][2]}{next_states[keys][0]}"] >= n:
           next_states.pop(keys)
        else: 
            hmap_pos[f"{next_states[keys][1]}{next_states[keys][2]}{next_states[keys][0]}"] = n

    if len(next_states) == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        return {0: [mean_y + n * 0.01 + calculate_complete_lines(grid) * 10]}

    # Exécution normale sans threads pour les autres profondeurs
    evaluer = {i: list(minimax_non_threaded(next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1, hmap_pos).values()) for i in next_states.keys()}


    if not 2 in evaluer.keys():
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        evaluer[2] = [[mean_y + n * 0.01 + calculate_complete_lines(grid) * 10]]


    # Retourne la valeur maximale pour chaque état
    return {i: max(evaluer[i]) for i in evaluer.keys()}

# Version avec threads pour la première profondeur
def minimax(grid, x, y, rot, n):
    hmap_pos = dict()

    if n == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        return {0: [mean_y]}
    
    next_states = get_next_states(grid, x, y, rot)

    for keys in next_states.keys():
        hmap_pos[f"{next_states[keys][1]}{next_states[keys][2]}{next_states[keys][0]}"] = n

    if len(next_states) == 0:
        return {0: [-999999]}

    # Paralléliser uniquement la première profondeur avec multiprocessing
    #with Pool() as pool:
    #    results = pool.starmap(minimax_non_threaded, [(next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1, hmap_pos) for i in reorder_list(next_states.keys())])

    evaluer = {i: list(minimax_non_threaded(next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1, hmap_pos).values()) for i in next_states.keys()}#{i: list(result.values()) for i, result in zip(next_states.keys(), results)}

    # Si la clé 2 n'existe pas dans evaluer
    if not 2 in evaluer.keys():
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
        evaluer[2] = [[mean_y + n * 0.01 + calculate_complete_lines(grid) * 10]]

    # Retourne la valeur maximale pour chaque état
    return {i: max(evaluer[i]) for i in evaluer.keys()}

#def minimax_from_states(state, n):
#    return minimax_non_threaded(state[3], state[1], state[2], state[0], n - 1)
