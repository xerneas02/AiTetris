import os
import numpy as np
from pyboy import PyBoy
from tensorflow.keras.models import load_model

from AccessMemory import get_grid_from_raw_screen, random_pieces, get_pos
from Rewards import get_game_reward, is_done
from Constante import action_space, num_actions, stop_action
from MemoryAdresse import ROTATION, ACTIVE_TETROMINO_Y
from CNN import create_cnn
from Minimax import *

# Constants
MODEL_NAME = "Model/2900_model.keras"
ROM_PATH = "Rom/Tetris.gb"
SHOW_DISPLAY = True
INPUT_SHAPE_GRID = (2, 18, 10)
FRAMES_PER_ACTION = 1
TEST_EPISODES = 10  # Number of test episodes

# Initialize PyBoy (GameBoy emulator)
pyboy = PyBoy(ROM_PATH, window_type="null" if not SHOW_DISPLAY else "SDL2")
pyboy.set_emulation_speed(0)  # Normal speed

# Load the trained model
if os.path.exists(MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Model loaded successfully.')
else:
    model = create_cnn(INPUT_SHAPE_GRID, num_actions)

model.summary()

# Function to test the model
def test_model():
    total_rewards = 0

    for episode in range(TEST_EPISODES):
        # Load initial game state
        with open("State/startstate.state", "rb") as f:
            pyboy.load_state(f)

        current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
        previous_grid = current_grid  # Initialize the grid

        done = False
        episode_reward = 0
        total_frames = 0
        frames = 0

        number_of_piece = 0

        current_x, current_y = get_pos(pyboy)
        last_x, last_y = current_x, current_y
        reset_off_set = False

        while not done:
            # Get the current game grid
            state = np.stack([previous_grid, current_grid])
    
            rot = pyboy.memory[ROTATION]

            if frames%53 == 52 and not reset_off_set:
                frames = 53 
                reset_off_set = True

            # Agent chooses an action
            # q_values = model.predict(np.array([state]), verbose=0)  # Predict action using the model
            result = minimax(current_grid, current_x, current_y, rot, 4)

            action_index = get_max_key(result) #np.argmax(q_values[0])  # Select action with the highest Q-value
            action = action_space[action_index]
            stop   = stop_action[action_index]

            if action_index != 2:
                print(frames%53)

            # Execute the action
            if pyboy.memory[ACTIVE_TETROMINO_Y] > 32:
                pyboy.send_input(action)
        
            
            previous_grid = current_grid

            current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)
            previous_grid_test = current_grid_test

            last_frame = total_frames
            while current_grid_test == previous_grid_test and not is_done(pyboy): 
                pyboy.tick()
                current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)
                frames += 1
            

            current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
            current_x, current_y = get_pos(pyboy)
            
            if current_y > last_y and action_index != 2:
                print("True")

            pyboy.send_input(stop)
            
            if current_y < last_y :
                random_pieces(pyboy)
                number_of_piece += 1
                frames = 0

            last_x, last_y = current_x, current_y

            # Calculate reward and check if game is done
            reward = get_game_reward(pyboy, current_grid, 0)
            
            if action_index == 2:
                frames = 0
            
            total_frames += 1
            episode_reward += reward

            done = is_done(pyboy)
                
            

        total_rewards += episode_reward
        print(f"Episode {episode + 1}, Score: {episode_reward/total_frames}")

    # Print the average reward after all test episodes
    print(f"Average reward after {TEST_EPISODES} test episodes: {total_rewards / TEST_EPISODES}")

# Run the test
test_model()

# Close the emulator after testing
pyboy.stop()
