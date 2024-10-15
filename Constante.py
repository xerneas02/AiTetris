from pyboy.utils import WindowEvent


ImageSize = 84
Ncouleur = 1
action_space = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_A]
stop_action = [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_A]
pieces_index = [0, 4, 8, 12, 16, 20, 24]
num_actions = len(action_space)