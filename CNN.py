from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def create_cnn(input_shape, num_actions):
    model = Sequential()
    
    # Input layer with shape (18, 10, 2) -> two grids stacked (current and previous)
    model.add(Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    
    # Flatten the convolutional layers output
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    
    # Output layer: One action for each possible move (e.g., num_actions moves)
    model.add(Dense(num_actions, activation='softmax'))
    
    return model
