import numpy as np
import random
import gym

class ThrottleEnv(gym.Env):

    def __init__(self):
        pass

    def _step(self, action):
        """
        :param
            -action: action to be taken at the timestep
        :return
            -ob (object): observations object
            -reward (float): reward from taking given action
            -episode_over (bool): True if episode is over, else False
            -info (dict): for debugging
        """
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        episode_over = False
        return ob, reward, episode_over, {}

    def _reset(self):
        pass

    def _render(self):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        return 0

    def _reset(self):
        pass

    def _get_state(self):
        return 0

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed


