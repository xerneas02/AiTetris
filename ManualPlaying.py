import pyboy.api
import pyboy.utils
from pynput import keyboard
from CNN import *
from DQNAgent import *

from tensorflow.keras.optimizers import Adam
from pyboy import PyBoy
from Rewards import *
import time
from Constante import *
import threading
from AccessMemory import *

from PIL import Image
import numpy as np

import os

def clear_console():
    # Check the current operating system
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')

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

# PyBoy ROM and settings
rom_path = "Rom/Tetris.gb"
show_display = True  # Set to True for real-time display

# Start PyBoy emulator
pyboy = PyBoy(rom_path)

if show_display:
    pyboy.set_emulation_speed(0)  # Real-time display
else:
    pyboy.set_emulation_speed(1_000_000)  # Fast mode without display

# Key mappings
key_mapping = {
    'z': 'up',
    'q': 'left',
    's': 'down',
    'd': 'right',
    'space': 'a',
    'shift': 'b',
    'a': 'start',
    'e': 'select',
    'r': 'save',  # "R" key for saving the game state
}


# Store the currently pressed keys
keys_pressed = set()

# Function to handle key press events
def on_press(key):
    try:
        if key.char in key_mapping:
            keys_pressed.add(key_mapping[key.char])
    except AttributeError:
        if key == keyboard.Key.shift:
            keys_pressed.add('b')
        elif key == keyboard.Key.space:
            keys_pressed.add('a')

# Function to handle key release events
def on_release(key):
    try:
        if key.char in key_mapping:
            keys_pressed.discard(key_mapping[key.char])
    except AttributeError:
        if key == keyboard.Key.shift:
            keys_pressed.discard('b')
        elif key == keyboard.Key.space:
            keys_pressed.discard('a')

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Function to save the current game state
def save_game_state(pyboy, filename="save_state.state"):
    with open(filename, "wb") as f:
        pyboy.save_state(f)
    print(f"Game state saved to {filename}")

# Main game loop for manual control
def play_manually():
    with open("State/startstate.state", "rb") as f:
        pyboy.load_state(f)
    total_frames = 0
    done = False
    reset_frame = False
    while not done:
        grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)

        if total_frames%53 == 52 or reset_frame:
            total_frames = 53
            reset_frame = False
        #clear_console()
        #print_grid(grid)
        #print(pyboy.memory[ROTATION])
        print(total_frames, total_frames%53)

        if show_display:
            time.sleep(0.016667)  # 60 FPS
        
        current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)
        previous_grid_test = current_grid_test

        # Check which keys are pressed and send the corresponding action to the game
        for action in keys_pressed:
            if action == 'save':  # Check if the save state button was pressed
                save_game_state(pyboy, "save_state.state")
            elif action == 'down':
                pyboy.button(action)
                reset_frame = True
            else:
                pyboy.button(action)

        x, y = get_pos(pyboy)
        last_y = y
        while (current_grid_test == previous_grid_test or y != get_pos(pyboy)[1]) and not is_done(pyboy): 
            pyboy.tick()
            current_grid_test = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy, False)
            x, y = get_pos(pyboy)
            total_frames += 1

        if y < last_y :
            total_frames = 0


# Start the manual control mode
play_manually()

listener.stop()  # Stop the keyboard listener when done
