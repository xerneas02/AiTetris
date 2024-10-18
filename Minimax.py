from AccessMemory import *
from Rewards import *
from multiprocessing import Pool

# Utility function to reorder the list based on a specific order
def reorder_list(tab):
    order = [0, 1, 3, 2]
    return [x for x in order if x in tab]

# Calculate evaluation score based on grid state
def calculate_evaluation_score(grid, x, y, rot, n):
    mean_y = sum([y + off_y for _, off_y in piece_rotation[rot]]) / 4
    holes_penalty = calculate_holes(grid) * 0.5
    lines_reward = calculate_complete_lines(grid) * 10
    position_penalty = abs(x - 5.5) * 0.01
    depth_penalty = n * 0.01
    return mean_y + lines_reward - holes_penalty + position_penalty + depth_penalty

# Handle base case of minimax
def base_case(grid, x, y, rot, n):
    return {0: [calculate_evaluation_score(grid, x, y, rot, n)]}

# Filter next states based on hash map position
def filter_next_states(next_states, n, hmap_pos):
    filtered_states = {}
    for key, state in next_states.items():
        state_key = f"{state[1]}{state[2]}{state[0]}"
        if state_key not in hmap_pos or hmap_pos[state_key] < n:
            hmap_pos[state_key] = n
            filtered_states[key] = state
    return filtered_states

# Recursive minimax function for deeper levels
def minimax_recursive(grid, x, y, rot, n, hmap_pos):
    if n == 0:
        return base_case(grid, x, y, rot, n)
    
    next_states = get_next_states(grid, x, y, rot)
    next_states = filter_next_states(next_states, n, hmap_pos)

    if not next_states:
        return base_case(grid, x, y, rot, n)
    
    evaluated_states = {i: list(minimax_recursive(
        next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1, hmap_pos
    ).values()) for i in reorder_list(next_states.keys())}

    # If key 2 does not exist in evaluated states, assign base case evaluation
    if 2 not in evaluated_states:
        evaluated_states[2] = [[calculate_evaluation_score(grid, x, y, rot, n)]]

    # Return the maximum evaluation for each state
    return {i: max(evaluated_states[i]) for i in evaluated_states.keys()}

# Main minimax function with threads for the first level
def minimax(grid, x, y, rot, n):
    if n == 0:
        return base_case(grid, x, y, rot, n)
    
    next_states = get_next_states(grid, x, y, rot)
    hmap_pos = {f"{next_states[key][1]}{next_states[key][2]}{next_states[key][0]}": n for key in next_states}

    if not next_states:
        return base_case(grid, x, y, rot, n)
    
    # Evaluate each next state non-threaded
    evaluated_states = {i: list(minimax_recursive(
        next_states[i][3], next_states[i][1], next_states[i][2], next_states[i][0], n - 1, hmap_pos
    ).values()) for i in reorder_list(next_states.keys())}

    # If key 2 does not exist in evaluated states, assign base case evaluation
    if 2 not in evaluated_states:
        evaluated_states[2] = [[calculate_evaluation_score(grid, x, y, rot, n)]]

    # Return the maximum evaluation for each state
    return {i: max(evaluated_states[i]) for i in evaluated_states.keys()}