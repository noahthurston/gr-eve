import numpy as np
import random
import gym

mod_to_index = {
    "bpsk": 0,
    "qpsk": 1,
    "8psk": 2,
    "16qam": 3
}

index_to_mod = {
    0: "bpsk",
    1: "qpsk",
    2: "8psk",
    3: "16qam"
}

mod_to_bps = {
    "bpsk": 1,
    "qpsk": 2,
    "8psk": 3,
    "16qam": 4
}

NUM_MODS = 4

class ThrottleEnv():

    def __init__(self):
        #np arr to keep track of last 4 success failures (rewarded if 2 of last 4 were failures)
        self.packet_record = np.zeros(NUM_MODS) + 1

        #current modulation, basically state
        self.curr_mod_index = 0

        #if packet succeeded, ==1
        self.packet_success = 1

    def print(self):
        print("self.packet_record: " + str(self.packet_record))
        print("self.curr_mod_index: " + str(self.curr_mod_index))
        print("self.packet_success: " + str(self.packet_success))

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
        reward = self._get_reward(action)

        ob = self._get_state()
        episode_over = False
        return ob, reward, episode_over

    def _render(self):
        pass

    def _take_action(self, action):
        #if jamming strength >=, packet fails
        if action >= 2:
            self.packet_success = 0
        else:
            self.packet_success = 1

        #update record
        self.packet_record = self.packet_record[:-3] + [self.packet_success]

        #check to go up/down modulation
        if np.sum(self.packet_record) <= 1:
            #drop modulation
            if self.curr_mod_index != 0: self.curr_mod_index -= 1
        elif np.sum(self.packet_record) >=3:
            #up modulation
            if self.curr_mod_index != 0: self.curr_mod_index += 1


    def _get_reward(self, action):
        #if packet was jammed, reward=3-jamming strength
        reward = 0
        if self.packet_success == 0:
            reward = 4 - action
        return reward

    def _reset(self):
        pass

    def _get_state(self):
        return self.curr_mod_index


