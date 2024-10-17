from AccessMemory import *
from Rewards import *

def print_grid(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 0:
                print(".", end="")
            if grid[i][j] == 1:
                print("#", end="")
            if grid[i][j] == 2:
                print("@", end="")
        print()

def add_to_list(liste, value):
    for i in range(len(liste)):
        liste[i] += value
    return liste

def minimax(grid, x, y, rot, n):
    if n == 0:
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]])/4
        return {0:[mean_y]}
    
    next_states = get_next_states(grid, x, y, rot)

    

    if len(next_states) == 0:
        return {0:[-999999]}

    evaluer = {i : list(minimax_from_states(next_states[i], n).values()) for i in next_states.keys()}

    if not 2 in evaluer.keys():
        mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]])/4
        evaluer[2] = [[mean_y+n*0.01+calculate_complete_lines(grid)*10]]

        


    return {i : max(evaluer[i]) for i in evaluer.keys()}


def minimax_from_states(state, n):
    return minimax(state[3], state[1], state[2], state[0], n-1)