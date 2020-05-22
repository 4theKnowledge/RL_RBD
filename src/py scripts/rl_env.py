# imports
import gym
from gym import spaces
import numpy as np
import random

# custom import
from des_env import des_env


class DiscreteEventSimEnv(gym.Env):
    """
    Define a simple discrete event simulator environment

    The environment defines which actions can be taken at which point and when the agent
    receives which reward.
    """

    def __init__(self):
        # General variables defining the environment
        self.TOTAL_TIME_STEPS = 5
        self.MIN_AVAILABILITY = 0.75

        self.curr_step = -1
        self.is_sim_finished = False

        # Define what the agent can do
        # The agent can:
        # 1. select the number of machines to use (dictated by system maximum, 10 - this will
        # be dependent on real life machine spares/availability)
        # 2. choose which type of machines to use (3 types)
        # ref:
        # https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
        # https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
        self.action_space = spaces.MultiDiscrete([10, 3])

        # Observation is the remaining time
        low = np.array([0.0])   # remaining_tries
        high = np.array([self.TOTAL_TIME_STEPS])    # remaining_tries
        self.observation_space = spaces.Box(low, high, dtype=np.float32)    # https://github.com/openai/gym/blob/master/gym/spaces/box.py

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action: int

        Returns
        ------

        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode was terminated. (For example,
                perhaps the pile tipped too far, or you lost your last life.)
            info (dict) :
                diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed
                to use this for learning.
        """

        if self.is_sim_finished:
            raise RuntimeError("Episode is done")

        self.curr_step += 1
        self._take_action(action)

        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_sim_finished, {}

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)

        if 0 < action[0]:   # the agent might choose zero machines and cause an issue in this section...
            def generate_machine_dict(action):
                """Generates dictionary of machines and their classes to pass to
                discrete event simulation environment"""

                machine_class_map = {0: 'A', 1: 'B', 2: 'C'}    # TODO: Change this to either or and get rid of map.

                machine_dict = {}
                for machine_no in range(action[0]):     # first col is the number of machines
                    machine_dict[machine_no] = machine_class_map[action[1]]
                return machine_dict
            # Generate machine dictionary
            machine_dict = generate_machine_dict(action)
            # Discrete Event Simulation
            des_env = des_env(machine_dict)

            # input agent action; return system availability
            self.sim_availability = des_env.availability
            sim_is_finished = self.MIN_AVAILABILITY < self.sim_availability

            if sim_is_finished:
                self.is_sim_finished = True

        self.remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = self.remaining_steps <= 0

        if time_is_over:    # Should this also be that if the agent has n consecutive low availabilities, it exits?
            self.is_sim_finished = True # abuse this a bit.


    def _get_reward(self):
            """Reward is given for getting a high availability within the least number of steps"""
            if self.is_sim_finished:
                # This can be figured out in the future...
                # Can incorporate costs, number of machines, etc etc.
                return self.sim_availability * self.remaining_steps
            else:
                return 0.0

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object) : the initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_sim_finished = False
        return self._get_state()

    def _render(self, mode="human", close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed


if __name__ == '__main__':
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

    history_string = "\n".join([f"Actions (Machines:{str(x[0][0])} | Types {str(x[0][1])})" for x in env.action_episode_memory])
    print(f'---History---\n{history_string}\n')


