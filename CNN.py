from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def create_cnn(input_shape, num_actions):
    model = Sequential()
    
    # Flatten the input grid
    model.add(Flatten(input_shape=input_shape))
    
    # Simple dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    # Output layer for each possible action
    model.add(Dense(num_actions, activation='linear'))  # Linear output for Q-values
    model.compile(optimizer='adam', loss='mse')  # Mean squared error for DQN
    
    return model
