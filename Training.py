import pyboy.api
import pyboy.utils
from CNN import *
from DQNAgent import *
from GameFrame import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from pyboy import PyBoy
from Rewards import *
import time
from Constante import *
import os

with open("rewards.log", "w") as f:
    f.write("")

model_name = "Model/model.keras"
rom_path = "Rom/Tetris.gb"
show_display = False

# Start PyBoy
pyboy = PyBoy(rom_path)
if show_display:
    pyboy.set_emulation_speed(0)
else:
    #pyboy = PyBoy(rom_path, window_type="headless")
    pyboy.set_emulation_speed(1_000_000)

# Define the input shape based on the grid (18, 10, 2)
input_shape_grid = (18, 10, 2)
num_actions = len(action_space)

if os.path.exists(model_name):
    model = load_model(model_name)
    print('Model loaded successfully')
else:
    model = create_cnn(input_shape_grid, num_actions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print('Model created successfully')

model.summary()

agent = DQNAgent(model, list(range(num_actions)), batch_size=512, epochs=1, verbose=0)

episodes = 1000000000000000000000
frames_per_action = 4
original_time = -1
train_time = 0
count = 0

for episode in range(episodes):
    with open("State/startstate.state", "rb") as f:
        pyboy.load_state(f)
    
    current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray)
    previous_grid = current_grid  # Initialize previous grid as the current grid initially

    total_frames = 0
    done = False
    
    while not done:
        random_pieces(pyboy)
        current_grid = get_grid_from_raw_screen(pyboy.screen.ndarray)

        # Stack the current grid and previous grid together as input
        state = np.stack([previous_grid, current_grid], axis=-1)
        
        if show_display:
            time.sleep(0.016667)
        
        if total_frames % frames_per_action == 0:
            action_index = agent.act(state)
            action = action_space[action_index]
        
        pyboy.button(action)
        pyboy.tick()
        
        reward = get_game_reward(pyboy)
        done = is_done(pyboy)
        
        next_grid = get_grid_from_raw_screen(pyboy.screen.ndarray)
        next_state = np.stack([current_grid, next_grid], axis=-1)
        
        if total_frames % frames_per_action == 0:
            agent.store_experience(state, action_index, reward, next_state, done)

        previous_grid = current_grid
        total_frames += 1
        count += 1 
        
    train_time = agent.train()
    model.save(model_name)

    reward = get_game_reward(pyboy)

    with open("rewards.log", "a") as f:
        f.write(f"{reward}\n")
    
    print(f"Episode {episode}, Score: {reward}")