import random
import numpy as np
import time
from collections import defaultdict, deque
import gc
import tensorflow as tf
from Constante import action_space as action_table



class DQNAgent:
    def __init__(self, model, action_space, batch_size=32, epochs=1, epsilon_stop_episode=2000, mem_size=1000, verbose=0):
        self.model = model
        self.action_space = action_space
        self.memory = deque(maxlen=mem_size)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Start fully exploring
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode  # Linear decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.action_count = defaultdict(int)
        self.verbose = verbose
        self.replay_start_size = 500  # Minimum experiences before training
        self.predict_fn = tf.function(self.model)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randint(0, self.action_space - 1)  # Explore
        else:
            # Use the compiled predict function
            q_values = self.predict_fn(np.array([state]))
            action = np.argmax(q_values[0])

        # Increment the count of the chosen action
        self.action_count[action] += 1

        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimized_train_step(self, states, q_values):
        self.model.fit(states, q_values, epochs=self.epochs, verbose=0)

    def train(self):
        if len(self.memory) < self.batch_size or len(self.memory) < self.replay_start_size:
            return
        
        start_time = time.time()
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        
        total_reward = 0
        for i in range(len(batch)):
            target = rewards[i]
            total_reward += rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(q_next[i])
            q_values[i][actions[i]] = target
        
        self.optimized_train_step(states, q_values)
        gc.collect()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay  # Linear decay

        elapsed_time = time.time() - start_time

        # Display total reward, time taken for training, and most used actions
        if self.verbose > 0: 
            print(f"Total reward = {total_reward}")
            print(f"Training time = {elapsed_time:.2f} seconds")
            
            # Calculate total number of actions
            total_actions = sum(self.action_count.values())

            if total_actions > 0:
                # Display the action usage as percentages
                print("Action usage percentage:")
                for action, count in sorted(self.action_count.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_actions) * 100
                    print(f"Action {action_table[action]}: {percentage:.2f}% ({count} times)")
            else:
                print("No actions taken during this training session.")
                
            print("------------------------------------------")
            
        return elapsed_time
    
    def save(self, model_name):
        self.model.save(model_name)
