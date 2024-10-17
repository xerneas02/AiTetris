from MemoryAdresse import *
from Constante import *
import random
from copy import deepcopy

def get_score(pyboy):
    # Retrieve the three bytes from memory
    byte1 = pyboy.memory[SCORE_BCD]     # C0A0
    byte2 = pyboy.memory[SCORE_BCD + 1] # C0A1
    byte3 = pyboy.memory[SCORE_BCD + 2] # C0A2

    # Convert from little-endian BCD to a decimal score
    score = (byte3 >> 4) * 100000 + (byte3 & 0x0F) * 10000
    score += (byte2 >> 4) * 1000 + (byte2 & 0x0F) * 100
    score += (byte1 >> 4) * 10 + (byte1 & 0x0F)

    return score


def get_pixel_color_from_raw_screen(screen_array, x, y):
    """
    Get the color of the pixel at (x, y) from the raw screen array.
    
    Parameters:
    - screen_array: The NumPy array of the unprocessed screen from pyboy.screen.ndarray.
    - x, y: The coordinates of the pixel.
    
    Returns:
    - A tuple representing the pixel color (R, G, B, A) for RGBA screens.
    """
    # Ensure the coordinates are within the screen bounds
    height, width, channels = screen_array.shape
    if x >= width or y >= height:
        raise ValueError(f"Coordinates ({x}, {y}) are out of bounds for this screen array.")

    # Return the pixel value at the given coordinates
    return tuple(screen_array[y, x])  # Access as (y, x) since NumPy uses row-major order

def is_in_bounds(tab, x, y):
    return 0 <= x < len(tab[0]) and 0 <= y < len(tab)

def get_grid_from_raw_screen(screen_array, pyboy, show = True):
    tab = [[0 for _ in range(10)] for _ in range(18)]  # Initialize an 18x10 grid
    x_start = 21  # Starting x-coordinate for the first cell
    y_start = 5   # Starting y-coordinate for the first cell
    size = 8      # Size of each block (in pixels)

    x = x_start
    y = y_start

    for i in range(len(tab)):
        for j in range(len(tab[i])):
            pixel_color = get_pixel_color_from_raw_screen(screen_array, x, y)
            # If the pixel is not white (255, 255, 255, 255 for RGBA), mark it as a block (1)
            if pixel_color != (255, 255, 255, 255):  # Adjust if your screen is not RGBA
                tab[i][j] = 1
            x += size  # Move to the next block horizontally
        x = x_start  # Reset x to the start of the next row
        y += size    # Move to the next block vertically
    
    if show:
        x, y = get_pos(pyboy)


        rot = pyboy.memory[ROTATION]

        draw_tetromino(tab, rot, x, y)
    
    return tab

def draw_tetromino(grid, rot, x, y, value = 2, verif = False):
    for i, j in piece_rotation[rot]:
        if is_in_bounds(grid, x+i, y+j) and grid[y+j][x+i] == 1 and verif:
            return None
        if is_in_bounds(grid, x+i, y+j):
            grid[y+j][x+i] = value
        elif verif:
            return None
    
    return grid

def random_pieces(pyboy):
    pyboy.memory[NEXT_TETROMINO_ADDRESS] = pieces_index[random.randint(0, len(pieces_index)-1)]

def get_pos(pyboy):
    x = int(((pyboy.memory[ACTIVE_TETROMINO_X]+1)/8)-4)
    y = int((pyboy.memory[ACTIVE_TETROMINO_Y]/8)-2)
    return x, y

def next_rotation(rotation):
    rotation += 1
    if rotation%4 == 0:
        rotation -= 4
    return rotation

def get_next_states(grid, x, y, rot):

    tab = deepcopy(grid)
    
    draw_tetromino(tab, rot, x, y, 0)

    next_states = dict()
    next_rot = next_rotation(rot)

    grid_tmp = draw_tetromino(deepcopy(tab), rot, x-1, y, verif=True)
    if grid_tmp: next_states[0] = (rot, x-1, y, grid_tmp)

    grid_tmp = draw_tetromino(deepcopy(tab), rot, x+1, y, verif=True)
    if grid_tmp: next_states[1] = (rot, x+1, y, grid_tmp)

    grid_tmp = draw_tetromino(deepcopy(tab), rot, x, y+1, verif=True)
    if grid_tmp: next_states[2] = (rot, x, y+1, grid_tmp)

    grid_tmp = draw_tetromino(deepcopy(tab), next_rot, x, y, verif=True)
    if grid_tmp: next_states[3] = (next_rot, x, y, grid_tmp)

    return next_states

def get_max_key(d):
    # Find the maximum value in the dictionary
    max_value = max(d.values())
    
    # Get all keys that have the maximum value
    max_keys = [key for key, value in d.items() if value == max_value]
    
    # Randomly select one key if there are multiple with the same max value
    return random.choice(max_keys)

def get_max_value(d):
    return max(d.values())