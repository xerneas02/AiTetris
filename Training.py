import os
import time
import numpy as np
from pyboy import PyBoy
from tensorflow.keras.models import load_model


from CNN import create_cnn
from DQNAgent import DQNAgent
from AccessMemory import get_grid_from_raw_screen, random_pieces
from Rewards import get_game_reward, is_done
from Constante import action_space, num_actions


# Constants
MODEL_NAME = "Model/model.h5"
ROM_PATH = "Rom/Tetris.gb"
SHOW_DISPLAY = True
INPUT_SHAPE_GRID = (2, 18, 10)
FRAMES_PER_ACTION = 4
EPISODES = 1_000_000_000_000_000_000_000
BATCH_SIZE = 32
EPOCHS = 3

# Clear the rewards log file
with open("rewards.log", "w") as f:
    f.write("")

# Initialize PyBoy (GameBoy emulator)
pyboy = PyBoy(ROM_PATH, window_type=None if not SHOW_DISPLAY else "SDL2")
pyboy.set_emulation_speed(1_000_000)

# Load or create model
if os.path.exists(MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Model loaded successfully.')
else:
    model = create_cnn(INPUT_SHAPE_GRID, num_actions)
    print('Model created successfully.')

model.summary()

# Initialize the DQN agent
agent = DQNAgent(model, num_actions, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)

# Main training loop
for episode in range(EPISODES):
    # Load the game state
    with open("State/startstate.state", "rb") as f:
        pyboy.load_state(f)

    # Initialize grids
    current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
    previous_grid = current_grid  # Set previous grid initially to current grid

    done = False
    total_frames = 0

    while not done:
        random_pieces(pyboy)

        # Update current grid
        current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)

        # Stack grids to form the state
        state = np.stack([previous_grid, current_grid])

        # Agent chooses an action
        action_index = agent.act(state)
        try:
            action = action_space[action_index]
        except Exception as e:
            print(f"Invalid action index: {action_index}")
            print(action_space)
            print(f"Error: {e}")
            exit(1)

        # Execute the action for a few frames
        for _ in range(FRAMES_PER_ACTION):
            pyboy.button(action)
            pyboy.tick()

        # Calculate reward and check if the game is over
        reward = get_game_reward(pyboy)
        done = is_done(pyboy)

        # Prepare next state and store experience in agent's memory
        next_grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)
        next_state = np.stack([current_grid, next_grid])
        agent.store_experience(state, action_index, reward, next_state, done)

        # Update previous grid for the next step
        previous_grid = current_grid
        total_frames += 1

    # Train the agent after each episode
    agent.train()

    # Save the model periodically
    try:
        agent.save(MODEL_NAME)
    except Exception as e:
        print(f"Error: Couldn't save the model. Reason: {e}")

    # Log the reward for the episode
    with open("rewards.log", "a") as f:
        f.write(f"{reward}\n")

    print(f"Episode {episode}, Score: {reward}")
