import pyboy.api
import pyboy.utils
from pynput import keyboard
from CNN import *
from DQNAgent import *
from GameFrame import *

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
    'p': 'screen'
}



def save_preprocessed_screen(pyboy, Ncouleur, filename):
    """
    Capture the preprocessed screen and the raw screen from PyBoy and save both as image files.
    
    Parameters:
    - pyboy: The instance of PyBoy emulator.
    - Ncouleur: Number of colors used in preprocess_frame (passed to your preprocess function).
    - filename: The name of the file to save the unprocessed screen (e.g., 'screen.png').
                The preprocessed screen will be saved as 'prepro_{filename}'.
    """
    # Preprocess the screen using the existing function
    preprocessed_screen = preprocess_frame(pyboy.screen.ndarray, Ncouleur)
    
    # Get the raw screen from PyBoy (original screen)
    raw_screen = pyboy.screen.ndarray


    print(f"Shape of preprocessed screen: {preprocessed_screen.shape}")
    print(f"Data type of preprocessed screen: {preprocessed_screen.dtype}")

    # Squeeze the extra dimension if it's grayscale with shape (84, 84, 1)
    if preprocessed_screen.shape[-1] == 1:
        preprocessed_screen = np.squeeze(preprocessed_screen, axis=-1)

    # Ensure both preprocessed and raw data are in uint8 format
    preprocessed_screen = np.uint8(preprocessed_screen * 255)  # Convert float64 to uint8 (assuming values are in [0, 1])
    raw_screen = np.uint8(raw_screen)  # Assuming raw screen values are already in [0, 255]

    # Save the preprocessed screen (assumed to be grayscale)
    if preprocessed_screen.ndim == 2:  # Grayscale image
        image_pre = Image.fromarray(preprocessed_screen, 'L')  # 'L' mode is for grayscale
    else:
        raise ValueError(f"Unsupported preprocessed screen shape: {preprocessed_screen.shape}")

    # Handle the raw screen (144, 160, 4) which is RGBA (with an alpha channel)
    if raw_screen.ndim == 3 and raw_screen.shape[-1] == 4:  # RGBA image
        image_raw = Image.fromarray(raw_screen, 'RGBA')  # Handle alpha channel
    else:
        raise ValueError(f"Unsupported raw screen shape: {raw_screen.shape}")

    # Save both images to files
    image_pre.save(f"prepro_{filename}")  # Save the preprocessed screen
    image_raw.save(filename)  # Save the raw screen with RGBA format
    
    print(f"Preprocessed screen saved as 'prepro_{filename}'")
    print(f"Raw screen saved as '{filename}'")


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
    while not done:
        grid = get_grid_from_raw_screen(pyboy.screen.ndarray, pyboy)

        clear_console()
        print_grid(grid)
        print(pyboy.memory[ROTATION])

        if show_display:
            time.sleep(0.016667)  # 60 FPS

        # Check which keys are pressed and send the corresponding action to the game
        for action in keys_pressed:
            if action == 'save':  # Check if the save state button was pressed
                save_game_state(pyboy, "save_state.state")
            if action == 'screen':  # Check if the save state button was pressed
                save_preprocessed_screen(pyboy, Ncouleur, "screen.png")
            else:
                pyboy.button(action)

        # Advance the game by one tick
        pyboy.tick()

        total_frames += 1


# Start the manual control mode
play_manually()

listener.stop()  # Stop the keyboard listener when done
