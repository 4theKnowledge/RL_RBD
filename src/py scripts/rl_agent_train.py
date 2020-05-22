"""
Training Environment for reinforcement learning agent.
Deep Q-Learning is used for this environment. However,
A3C will be used in the future.

This script utilises keras-rl API:
https://github.com/keras-rl/keras-rl/tree/master/examples

@author: Tyler
"""

# imports
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# custom imports
from des_env import DiscreteEventSimEnv

def agent_run():
    """
    Trains a RL agent to learn the environment.
    """
    pass

def manual_run():
    """
    Performs a test of the system/environment using manually specified parameters.
    """
    # initialise environment
    env = DiscreteEventSimEnv()

    # Run for n episodes
    no_episodes = 1
    print(f'Running for {no_episodes} episodes')
    for episode in range(no_episodes):
        # reset environment
        env.reset()
        while not env.is_sim_finished:
            # take random action
            random_action = env.action_space.sample()
            print(f'Random action {random_action}')
            ob, reward, _, _ = env.step(random_action)
            print(f'Observation (time steps remaining): {ob[0]}\nReward: {reward}\n')

        print(f'Sim finished? {env.is_sim_finished}\n')

    history_string = "\n".join(
        [f"Actions (Machines:{str(x[0][0])} | Types {str(x[0][1])})" for x in env.action_episode_memory])
    print(f'---History---\n{history_string}\n')


class DiscreteEventSimProcessor(Processor):
    def process_observation(self, observation):
        pass
    def process_state_batch(self, batch):
        pass
    def process_reward(self, reward):
        pass


OBSERVATION_SPACE_SIZE = 1
ACTION_SPACE_SIZE = 3   # How do we incorporate [n,m] action space into the output layer of a NN?

# Get the environment and extract the number of actions
env = DiscreteEventSimEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape     # TODO: review how this works with the MultiDiscrete
print(nb_actions)

# Next, we build the model. This model takes in the observations and outputs the actions.
# nn code: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# Our observation space only consists of the number of TIME_STEPS remaining
# whereas the actions are the number machines and their class
# TODO: Review how the engineer the observation space to make it more feature rich

model = Sequential()
model.add(Dense(8, input_dim=OBSERVATION_SPACE_SIZE, activation='relu'))
model.add(Dense(ACTION_SPACE_SIZE, activation='softmax'))
print(model.summary())

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1_000)   # this was 1_000_000

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)





if __name__ == '__main__':
    # manual_run()
    pass