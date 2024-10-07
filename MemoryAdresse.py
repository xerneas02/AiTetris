# addresses from https://datacrystal.tcrf.net/wiki/Tetris_(Game_Boy)/RAM_map

SCORE_BCD = 0xC0A0
SCORES_B = 0xD000
SCORES_A = 0xD654

NEW_HIGHSCORE_BOOL = 0xFFE8

GAME_STATE = 0xFFE1 #13 perdu / 4 Ecran Game over / 0 In game  
P1_HEIGHT = 0xFFAD


SCORE_MOD_PTR_HI_MAYBE = 0xFFFB
SCORE_MOD_PTR_LO_MAYBE = 0xFFFC

STATIC_BLOCKS_START_ADDRESS = 0xC800  # Adresse des blocs statiques
NEXT_TETROMINO_ADDRESS = 0xC213     # Adresse du tetromino actif
ACTIVE_TETROMINO_X = 0xC200           # Position X du tetromino actif
ACTIVE_TETROMINO_Y = 0xC201           # Position Y du tetromino actif
ROTATION = 0xC203

Y = 0xFFB2
X = 0xFFB3