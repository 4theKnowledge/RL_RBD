"""
DQN model - from scratch
ref: pythonprogramming.net / sentdex

@author: Tyler
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

import numpy as np
import time
import random

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64     # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5     # terminal states (end of episodes)
OBSERVATION_SPACE_SIZE = 1
ACTION_SPACE_SIZE = 3  # How do we incorporate [n,m] action space into the output layer of a NN?


class DQNAgent:
    def __init__(self):
        # main model which gets trained at every step
        self.model = self.create_model()

        # Target model - this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        # self.tensorboard

        # Used to coutn when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=OBSERVATION_SPACE_SIZE, activation='relu'))
        model.add(Dense(ACTION_SPACE_SIZE, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001),metrics=['accuracy'])
        return model

    # Adds steps' data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # get minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.target_model.predict(current_states)

        # Get future states from minibatch , then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q learning, but we just use part of the equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False)    # TODO: add callbacks

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if UPDATE_TARGET_EVERY < self.target_update_counter:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]