from MemoryAdresse import *
from Constante import *
import random

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

def get_grid_from_raw_screen(screen_array, pyboy):
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
    
    x, y = get_pos(pyboy)
    
    tab[y][x] = 2
    rot = pyboy.memory[ROTATION]
    #L Right
    if rot == 0:
        x, y = x-1, y+1
        tab[y][x] = 2
        tab[y-1][x]   = 2
        tab[y-1][x+1] = 2
        tab[y-1][x+2] = 2
    
    if rot == 1:
        x, y = x+1, y+1
        tab[y][x] = 2
        tab[y][x-1]   = 2
        tab[y-1][x-1] = 2
        tab[y-2][x-1] = 2
        
    if rot == 2:
        x, y = x+1, y
        tab[y][x] = 2
        tab[y-1][x] = 2
        tab[y][x-1] = 2
        tab[y][x-2] = 2
        
    if rot == 3:
        x, y = x, y+1
        tab[y][x] = 2
        tab[y-1][x]   = 2
        tab[y-2][x]   = 2
        tab[y-2][x-1] = 2
        
    #L Left
    if rot == 4:
        tab[y][x-1]   = 2
        tab[y][x+1]   = 2
        tab[y+1][x+1] = 2
    
    if rot == 5:
        tab[y-1][x]   = 2
        tab[y-1][x+1] = 2
        tab[y+1][x]   = 2
        
    if rot == 6:
        tab[y][x+1] = 2
        tab[y][x-1] = 2
        tab[y-1][x-1] = 2
        
    if rot == 7:
        tab[y-1][x]   = 2
        tab[y+1][x]   = 2
        tab[y+1][x-1] = 2
    
    #Line Piece 
    if rot == 8 or rot == 10: 
        for i in range(4):
            tab[y][x+i-1] = 2
            
    if rot == 9 or rot == 11:
        for i in range(4):
            tab[y-i+1][x] = 2
            
    #Square
    if rot >= 12 and rot <= 15:
        tab[y+1][x]   = 2
        tab[y][x+1]   = 2
        tab[y+1][x+1] = 2
    
    #Zigzag Left
    if rot == 18 or rot == 16:
        tab[y][x-1]   = 2
        tab[y+1][x]   = 2
        tab[y+1][x+1] = 2
    
    if rot == 17 or rot == 19:
        tab[y][x-1]   = 2
        tab[y-1][x]   = 2
        tab[y+1][x-1] = 2
    
    #Zigzag Right
    if rot == 20 or rot == 22:
        tab[y][x+1]   = 2
        tab[y+1][x]   = 2
        tab[y+1][x-1] = 2
    
    if rot == 21 or rot == 23:
        tab[y][x-1]   = 2
        tab[y+1][x]   = 2
        tab[y-1][x-1] = 2
    
    #  @@@
    #   @
    if rot == 24:
        tab[y][x+1] = 2
        tab[y][x-1] = 2
        tab[y+1][x] = 2

    if rot == 25:
        tab[y][x+1] = 2
        tab[y-1][x] = 2
        tab[y+1][x] = 2
        
    if rot == 26:
        tab[y][x+1] = 2
        tab[y][x-1] = 2
        tab[y-1][x] = 2
        
    if rot == 27:
        tab[y][x-1] = 2
        tab[y-1][x] = 2
        tab[y+1][x] = 2
    
    return tab

def random_pieces(pyboy):
    pyboy.memory[NEXT_TETROMINO_ADDRESS] = pieces_index[random.randint(0, len(pieces_index)-1)]

def get_pos(pyboy):
    x = int(((pyboy.memory[ACTIVE_TETROMINO_X]+1)/8)-4)
    y = int((pyboy.memory[ACTIVE_TETROMINO_Y]/8)-2)
    

    #x = x if x >= 0 else 0
    #x = x if x < 10 else 9

    return x, y