import os
import numpy as np
import pandas as pd  # Ajout de pandas pour gérer les données
from pyboy import PyBoy

from AccessMemory import get_grid_from_raw_screen, random_pieces, get_pos
from Rewards import get_game_reward, is_done
from Constante import action_space, stop_action
from MemoryAdresse import ROTATION, ACTIVE_TETROMINO_Y
from Minimax import *

# Constants
ROM_PATH = "Rom/Tetris.gb"
SHOW_DISPLAY = True
TEST_EPISODES = 1000  # Number of test episodes
DEPTH = 18
DATA_SAVE_PATH = "data/minimax_data.csv"  # Chemin du fichier pour sauvegarder les données

# Initialize PyBoy (GameBoy emulator)
pyboy = PyBoy(ROM_PATH, window_type="null" if not SHOW_DISPLAY else "SDL2")
pyboy.set_emulation_speed(0)  # Normal speed

# Function to reset the game state
def reset_game_state():
    with open("State/startstate.state", "rb") as f:
        pyboy.load_state(f)

# Function to initialize episode variables
def initialize_episode():
    current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
    previous_grid = current_grid
    current_x, current_y = get_pos(pyboy)
    return current_grid, previous_grid, current_x, current_y

# Function to save state-action pairs
def save_state_action(state, action):
    # Vérifie si state est déjà un tableau NumPy
    if not isinstance(state, np.ndarray):
        state = np.array(state)  # Conversion en tableau NumPy si nécessaire
    
    state_flat = state.flatten()  # Flatten the grid to a 1D array
    data = list(state_flat) + [action]  # Concatenate state and action
    
    # Append the data to the CSV file
    df = pd.DataFrame([data])
    df.to_csv(DATA_SAVE_PATH, mode='a', header=not os.path.exists(DATA_SAVE_PATH), index=False)


# Function to perform a step in the game
def perform_game_step(current_grid, previous_grid, current_x, current_y, random_piece_count):
    state = np.stack([previous_grid, current_grid])
    rot = pyboy.memory[ROTATION]
    
    # Use Minimax to determine the next action
    result = minimax(current_grid, current_x, current_y, rot, DEPTH)
    action_index = get_max_key(result)
    action = action_space[action_index]
    stop = stop_action[action_index]

    # Save the state and the chosen action
    save_state_action(current_grid, action_index)

    # Execute the action
    if action_index == 3:
        pyboy.tick()

    if pyboy.memory[ACTIVE_TETROMINO_Y] > 32:
        pyboy.send_input(action)
    
    return action_index, stop

# Function to run a test episode
def run_test_episode():
    episode_reward = 0
    total_frames = 0
    random_piece_count = 0
    current_grid, previous_grid, current_x, current_y = initialize_episode()

    while not is_done(pyboy):
        action_index, stop = perform_game_step(
            current_grid, previous_grid, current_x, current_y, random_piece_count
        )
        
        # Update the grid and position after action execution
        current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)
        previous_grid_test = current_grid_test

        while current_grid_test == previous_grid_test and not is_done(pyboy):
            pyboy.tick()
            current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)

        previous_grid = current_grid
        current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
        current_x, current_y = get_pos(pyboy)

        # Handle piece change
        if current_y < pyboy.memory[ACTIVE_TETROMINO_Y]:
            random_pieces(pyboy)
            random_piece_count += 1
            

        # Calculate reward and accumulate for the episode
        reward = get_game_reward(pyboy, current_grid, 0)
        episode_reward += reward
        pyboy.send_input(stop)
        total_frames += 1

    return get_score(pyboy)

# Main function to test the Minimax algorithm
def test_minimax():
    total_rewards = 0
    for episode in range(TEST_EPISODES):
        reset_game_state()
        episode_score = run_test_episode()
        total_rewards += episode_score
        print(f"Episode {episode + 1}, Score: {episode_score}")
    
    average_reward = total_rewards / TEST_EPISODES
    print(f"Average reward after {TEST_EPISODES} test episodes: {average_reward}")

# Run the test
test_minimax()

# Close the emulator after testing
pyboy.stop()
