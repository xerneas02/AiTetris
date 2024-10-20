import os
import numpy as np
import pandas as pd
import tensorflow as tf  # Import for loading the trained model
from pyboy import PyBoy

from AccessMemory import get_grid_from_raw_screen, random_pieces, get_pos, get_score
from Rewards import get_game_reward, is_done
from Constante import action_space, stop_action
from MemoryAdresse import ACTIVE_TETROMINO_Y

# Constants
ROM_PATH = "Rom/Tetris.gb"
SHOW_DISPLAY = True
MODEL_PATH = "Model/tetris_minimax_model.h5"  # Path to the saved trained model
DEPTH = 18

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

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

# Function to preprocess the grid (state) for model prediction
def preprocess_grid(grid):
    # Flatten the grid and reshape it to fit the model's input format
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    grid_flat = grid.flatten()
    return np.expand_dims(grid_flat, axis=0)  # Add a batch dimension for prediction

# Function to perform a step using the model to predict the action
def perform_model_step(current_grid, previous_grid, current_x, current_y):
    state = np.stack([previous_grid, current_grid])
    
    # Preprocess the current grid for model input
    processed_grid = preprocess_grid(current_grid)

    # Use the model to predict the next action
    predictions = model.predict(processed_grid)
    action_index = np.argmax(predictions)  # Get the action with the highest probability
    action = action_space[action_index]
    stop = stop_action[action_index]

    # Execute the action
    if action_index == 3:
        pyboy.tick()

    if pyboy.memory[ACTIVE_TETROMINO_Y] > 32:
        pyboy.send_input(action)
    
    return action_index, stop

# Function to run a test episode with the trained model
def run_model_episode():
    episode_reward = 0
    total_frames = 0
    random_piece_count = 0
    current_grid, previous_grid, current_x, current_y = initialize_episode()

    while not is_done(pyboy):
        action_index, stop = perform_model_step(
            current_grid, previous_grid, current_x, current_y
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

# Main function to run the model and let it play
def test_model():
    total_rewards = 0
    TEST_EPISODES = 10  # Number of test episodes
    for episode in range(TEST_EPISODES):
        reset_game_state()
        episode_score = run_model_episode()
        total_rewards += episode_score
        print(f"Episode {episode + 1}, Score: {episode_score}")
    
    average_reward = total_rewards / TEST_EPISODES
    print(f"Average reward after {TEST_EPISODES} test episodes: {average_reward}")

# Run the test
test_model()

# Close the emulator after testing
pyboy.stop()
