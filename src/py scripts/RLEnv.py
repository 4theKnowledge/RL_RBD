"""
Interactive environment for RL agents to interact with the DES environment.
see: https://github.com/openai/gym/blob/master/gym/core.py
@author: Tyler Bikaun

"""


class InteractiveEnv(object):
    """
    Encapsulates an environment with arbitrary behind-the-scenes dynamics.

    The main API methods that users of this class need to know are:

        step
        reset
        close
        seed

    And the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    """
    def __init__(self):
        pass

    def step(self):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info)

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step()
                        calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging
                         and sometimes learning)
        """

        pass

    def reset(self):
        """
        Resets the state of the environment and returns an initial observations.

        Returns:
            observation (object) : the initial observation.
        """
        # Reset to a default system architecture (DES)?
        pass

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the "main"
            seed, or the value which a reproducer should pass to 'seed'. Often,
            the main seed equals the provided 'seed', but this won't be true if
            seed=None, for example.
        """
        pass