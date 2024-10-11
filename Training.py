import os
import numpy as np
from pyboy import PyBoy
from tensorflow.keras.models import load_model

from CNN import create_cnn
from DQNAgent import DQNAgent
from AccessMemory import get_grid_from_raw_screen, random_pieces
from Rewards import get_game_reward, is_done
from Constante import action_space, num_actions, stop_action
from MemoryAdresse import ROTATION, ACTIVE_TETROMINO_Y

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Constants
MODEL_NAME = "Model/model.keras"
ROM_PATH = "Rom/Tetris.gb"
SHOW_DISPLAY = False
INPUT_SHAPE_GRID = (2, 18, 10)
FRAMES_PER_ACTION = 1  # Reduced repetition
EPISODES = 6000
BATCH_SIZE = 128
EPOCHS = 1

# Clear the rewards log file
with open("rewards.log", "w") as f:
    f.write("")

# Initialize PyBoy (GameBoy emulator)
pyboy = PyBoy(ROM_PATH, window_type="null" if not SHOW_DISPLAY else "SDL2")
pyboy.set_emulation_speed(0)  # Normal speed

# Load or create model
if os.path.exists(MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Model loaded successfully.')
else:
    model = create_cnn(INPUT_SHAPE_GRID, num_actions)
    print('Model created successfully.')

model.summary()

# Initialize the DQN agent
agent = DQNAgent(model, num_actions, batch_size=BATCH_SIZE, epochs=EPOCHS, epsilon_stop_episode=EPISODES/2, mem_size=1000)

# Main training loop
for episode in range(EPISODES):
    with open("State/startstate.state", "rb") as f:
        pyboy.load_state(f)

    current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
    previous_grid = current_grid  # Set previous grid initially to current grid

    done = False
    
    total_reward = 0
    total_input = 0
    
    while not done:
        random_pieces(pyboy)
        state = np.stack([previous_grid, current_grid])
        
        action_index = agent.act(state)
        action = action_space[action_index]
        stop   = stop_action[action_index]

        
        if pyboy.memory[ACTIVE_TETROMINO_Y] > 32:
            pyboy.send_input(action)
        
        previous_grid = current_grid
        while previous_grid == current_grid: 
            pyboy.tick()
            current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
        pyboy.send_input(stop)


        reward = get_game_reward(pyboy, current_grid, total_input)
        done = is_done(pyboy)
        
        reward += -20 if done else 0
        
        total_reward += reward
        total_input  += 1

        next_state = np.stack([previous_grid, current_grid])
        agent.store_experience(state, action_index, reward, next_state, done)


    agent.train()

    # Save the model periodically
    if episode % 100 == 0:
        agent.save(MODEL_NAME)

    print(f"Episode {episode}, Score: {total_reward/total_input}")
    
    with open("rewards.log", "a") as f:
        f.write(f"{total_reward/total_input}\n")
        
