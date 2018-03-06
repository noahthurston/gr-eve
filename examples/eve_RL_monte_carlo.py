#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve Re Learn Testbed Graph
# Generated: Tue Jan 23 11:09:53 2018
##################################################

"""
if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
"""
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy as np
import sys
from gnuradio import qtgui

import bitarray
import tables
from tables import *
import matplotlib.pyplot as plt
import random

class eve_re_learn_testbed_graph(gr.top_block):

    def __init__(self, pattern_as_vector, eve_noise_db=1, channel_noise_db=1, max_items=4):
        gr.top_block.__init__(self, "Eve Re Learn Testbed Graph")

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 0
        self.samp_rate = samp_rate = 10000000
        self.max_items = max_items
        self.eve_noise_db = eve_noise_db
        self.channel_noise_db = channel_noise_db
        self.const = const = digital.constellation_calcdist(([-1-1j, -1+1j, 1+1j, 1-1j]), ([0, 1, 3, 2]), 4, 1).base()

        ##################################################
        # Blocks
        ##################################################
        self.blocks_vector_source_x_0 = blocks.vector_source_c(pattern_as_vector, False, 1, [])

        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(const)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc((const.points()), 1)
        self.blocks_vector_sink_alice = blocks.vector_sink_b(1)
        self.blocks_vector_sink_bob = blocks.vector_sink_b(1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_repack_bits_bb_1 = blocks.repack_bits_bb(1, 8, "", False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(1, const.bits_per_symbol(), "", False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(const.bits_per_symbol(), 8, "", False, gr.GR_LSB_FIRST)
        self.blocks_head_0_0 = blocks.head(gr.sizeof_char*1, max_items)
        self.blocks_head_0 = blocks.head(gr.sizeof_char*1, max_items)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, np.random.randint(0, 4, 1000000)), True)
        #self.analog_fastnoise_source_x_0_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(self.eve_noise_db/20.0), 0, 2**16)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(self.channel_noise_db/20.0), 0, 2**16)

        #self.blocks_vector_source_x_1 = blocks.vector_source_b((0,0), True, 1, [])

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        #self.connect((self.analog_fastnoise_source_x_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_add_xx_0, 2))

        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_1, 0))

        #self.connect((self.blocks_vector_source_x_1, 0), (self.blocks_repack_bits_bb_0_0, 0))
        #self.connect((self.blocks_vector_source_x_1, 0), (self.blocks_repack_bits_bb_1, 0))

        self.connect((self.blocks_add_xx_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_vector_sink_bob, 0))
        self.connect((self.blocks_head_0_0, 0), (self.blocks_vector_sink_alice, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_repack_bits_bb_1, 0), (self.blocks_head_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.blocks_repack_bits_bb_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "eve_re_learn_testbed_graph")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.analog_fastnoise_source_x_0_0.set_amplitude(10**(-self.snr_db/20.0))
        self.analog_fastnoise_source_x_0.set_amplitude(10**(-self.snr_db/20.0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_max_items(self):
        return self.max_items

    def set_max_items(self, max_items):
        self.max_items = max_items
        self.blocks_head_0_0.set_length(self.max_items)
        self.blocks_head_0.set_length(self.max_items)

    def get_const(self):
        return self.const

    def set_const(self, const):
        self.const = const

# main function generated by grc
def main(top_block_cls=eve_re_learn_testbed_graph, options=None):

    """
    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)
    """

    tb = top_block_cls()
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


"""
New idea
-jam at random points
-say the average reward at those points is the average reward of every set its participated in
-during explore: pick random 4 points to block
-during exploit: pick top 4 performing points to block

pros:
-could find best set of bytes to block
-wouldn't be bound by static "patterns"

cons:
-very dependent on fixed packet sizes (but could later move to state dependent reinforcement learning algorithms) 
"""

#created separate class so that the reinforcement learning table could be accessed easily anywhere within the function
class monte_carlo_model():
    def __init__(self, arms, arm_counts, arm_rewards, arm_average_rewards, num_episodes, artificial_rewards, artificial_best_pattern, power_penalty_list):
    #def __init__(self, epsilon, eve_noise_patterns, arm_counts, average_rewards, bytes_per_packet, with_power_penalty, bits_flipped_threshold):
        
        """
        self.epsilon = epsilon
        self.eve_noise_arms = eve_noise_arms #list of jamming patterns
        self.arm_counts = arm_counts #counts of how many times each arm has been taken
        self.average_rewards = average_rewards
        self.bytes_per_packet = bytes_per_packet 
        self.with_power_penalty = with_power_penalty
        #number of bits to flip in order to get a reward
        self.bits_flipped_threshold = bits_flipped_threshold
        #record of average rewards for debugging
        self.historical_average_rewards = [[]]
        self.historical_pickaction_choices = [[]]
        """

        #MONTE CARLO
        self.arms = arms
        self.arm_counts = arm_counts
        self.arm_rewards = arm_rewards
        self.arm_average_rewards = arm_average_rewards

        self.curr_episode = 0
        self.bytes_per_part = 1
        self.parts_per_packet = len(arms)

        #artificial rewards by timesteps that the algorithm is going to try to model
        self.artificial_rewards = artificial_rewards
        self.artificial_best_pattern = artificial_best_pattern
        self.power_penalty_list = power_penalty_list

        #current pattern and result used when running episodes
        self.curr_pattern = np.zeros((len(arm_counts)))
        self.curr_part = 0

        #array of ints that were recieved sent/rec
        self.total_alice_sent_int = np.array([])
        self.total_bob_rec_int = np.array([])

        #array of ints counting how many bits were flipped per byte
        self.bits_flipped_by_byte = np.array([])


        self.episode_reward = 0

        self.average_reward_per_arm = np.zeros((len(self.arm_counts),len(self.arm_counts[0])))
        self.best_pattern = np.zeros((len(self.arm_counts)),dtype=int)

        self.pattern_mistakes_over_time = np.array([len(self.arm_counts)])

        self.update_graph_data_increment = 10



    # main function, iterates through trials 
    def train_model(self, num_trials):
        print("beginning training")

        """
        self.historical_average_rewards = [[0 for x in range(len(self.eve_noise_arms))] for y in range(num_trials)]
        self.historical_pickaction_choices = [[0 for x in range(len(self.eve_noise_arms))] for y in range(num_trials)]

        # loop through trainning trialss
        for trial in range(num_trials):
            print("\nrunning trial #%d" %trial)
            self.run_trial() 

            print("current rewards: "+ str(self.average_rewards))
            total_counts = float(sum(self.arm_counts))
            print("current pickation frequency: "+ str([float(count)/total_counts for count in self.arm_counts]))
            
            # recording values for graphing average rewards over time
            for index, avg in enumerate(self.average_rewards):
                self.historical_average_rewards[trial][index] = avg

            # recorddng values for graphing the percentage of each arm chosen over time
            for index, count in enumerate(self.arm_counts):
                self.historical_pickaction_choices[trial][index] = (float(count)/float(sum(self.arm_counts)))

            #print(self.historical_average_rewards)

        self.graph_averages_overtime()
        self.graph_pickaction_choices_overtime()
        """

        #MONTE CARLO
        #print(self.artificial_rewards)

        while self.curr_episode < num_episodes:
            print("EPISODE #" + str(self.curr_episode) + "-------------------------------------")
            self.run_episode()
            print("\n")
            """
            if(self.curr_episode % self.update_graph_data_increment == 0):
                self.calculate_average_reward_per_arm()
                print(self.average_reward_per_arm)
                self.calculate_best_pattern()
                self.update_graph_data()
            """

            self.curr_episode = self.curr_episode + 1

        #self.calculate_average_reward_per_arm()
        
        #print(self.average_reward_per_arm)

        #self.calculate_best_pattern()
        #print("END, best pattern: " + str(self.best_pattern))

        #self.graph_pattern_mistakes_over_time()

        print("done training")




    #runs through whole episode process
    def run_episode(self):
        self.curr_part = 0
        self.generate_random_pattern()
        self.run_through_model()
        self.calculate_reward()
        self.update_model()
        self.calculate_best_pattern()

        print("self.arm_average_rewards: \n" + str(self.arm_average_rewards))



    """
    # EG function
    # given the current state and inputs, it calculates the action to be taken
    def pick_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = self.average_rewards.index(max(self.average_rewards))
        else:
            chosen_arm = random.randrange(len(self.eve_noise_arms))

        print("chosen_arm:%d , eve noise db:%d" % (chosen_arm, self.eve_noise_arms[chosen_arm]))
        return chosen_arm
    """


    #generates random pattern
    def generate_random_pattern(self):
        for time, curr_arms in enumerate(self.arms):
            self.curr_pattern[time] = random.randint(0,len(curr_arms)-1)


    def pattern_to_vector(self):
        #need to find real number
        samples_per_byte = 4

        pattern_as_vector = np.zeros((samples_per_byte*self.parts_per_packet))

        for pattern_index, pattern_part in enumerate(self.curr_pattern):
            for vector_index in range(pattern_index*samples_per_byte, (pattern_index+1)*samples_per_byte):
                # set vector value to sample from gaussian distribution
                pattern_as_vector[vector_index] = np.random.normal(0,10**(self.arms[pattern_index][int(pattern_part)]/20.0))

        print("pattern_as_vector: " + str(pattern_as_vector))
        return pattern_as_vector 


    # using generated pattern, runs episode through the model
    def run_through_model(self):
        #right now this done nothing, but it will be used when sending signals through GNU radio
        print("self.curr_pattern: " + str(self.curr_pattern))


        self.total_alice_sent_int = np.array([])
        self.total_bob_rec_int = np.array([])

        pattern_as_vector = self.pattern_to_vector()

        #for timestep, chosen_arm_index in enumerate(self.curr_pattern):
        
        #print("chosen_noise_db: " + str(self.arms[self.curr_part][int(chosen_arm_index)]))
        tb = eve_re_learn_testbed_graph(pattern_as_vector, eve_noise_db=0, channel_noise_db=-100, max_items=4)
        tb.start()
        
        tb.wait()


        #save data to calculate reward later
        if len(self.total_alice_sent_int) == 0:
            self.total_alice_sent_int = np.array(tb.blocks_vector_sink_alice.data())
            self.total_bob_rec_int = np.array(tb.blocks_vector_sink_bob.data())
        else:
            self.total_alice_sent_int = np.append(self.total_alice_sent_int, np.array(tb.blocks_vector_sink_alice.data()))
            self.total_bob_rec_int = np.append(self.total_bob_rec_int, np.array(tb.blocks_vector_sink_bob.data()))

            #tb.stop()
        #print("self.total_alice_sent: " + str(self.total_alice_sent))
        #print("self.total_bob_rec: " + str(self.total_bob_rec))


    # given what alice sent and bob received, it scores Eve's decision
    def calculate_reward(self):
        self.calculate_bits_flipped_per_byte()
        print("self.bits_flipped_by_byte: " + str(self.bits_flipped_by_byte))
        self.episode_reward = 0

        for index, bits_flipped in enumerate(self.bits_flipped_by_byte):
            #rewarding for flipping at least 2 bits in the byte
            if bits_flipped >= 2:
                self.episode_reward=(self.episode_reward+self.artificial_rewards[index])
                print("adding: "+ str(self.artificial_rewards[index]))
            else:
                print("adding: 0")

            #power penalty
            self.episode_reward=(self.episode_reward+self.power_penalty_list[int(self.curr_pattern[index])])
            print("penalty: " + str(self.power_penalty_list[int(self.curr_pattern[index])]))

        print("self.episode_reward: " + str(self.episode_reward))


        #MONTE CARLO
        #self.episode_reward = 0
        #for timestep, arm_chosen in enumerate(self.curr_pattern):
        #    self.episode_reward = self.episode_reward + self.artificial_rewards[int(timestep)][int(arm_chosen)]



    def calculate_value_per_arm(self):
        for timestep in range(len(self.arms)):
            for arm_index in range(len(self.arms[timestep])):
                if(self.arm_counts[timestep][arm_index] != 0):
                    self.arm_average_rewards[timestep][arm_index] = float(self.arm_rewards[timestep][arm_index]) / float(self.arm_counts[timestep][arm_index])


    def calculate_best_pattern(self):
        for timestep in range(len(self.best_pattern)):
            self.best_pattern[timestep] = self.arm_average_rewards[timestep].argmax(axis=0)
            #print("argmax: " + str(self.arm_average_rewards[timestep].argmax(axis=0)))

        print("self.best_pattern: " + str(self.best_pattern))



    ### needs works
    def update_model(self):
        #distribute reward evenly among all arms taken within the pattern
        num_timesteps = len(self.arms)
        reward_per_arm = self.episode_reward / num_timesteps

        for timestep in range(num_timesteps):
            chosen_arm_index = self.curr_pattern[timestep]
            self.arm_rewards[timestep][int(chosen_arm_index)]  = self.arm_rewards[timestep][int(chosen_arm_index)] + reward_per_arm
            self.arm_counts[timestep][int(chosen_arm_index)] = self.arm_counts[timestep][int(chosen_arm_index)] + 1

        self.calculate_value_per_arm()





    def calculate_bits_flipped_per_byte(self):
        self.bits_flipped_by_byte = np.array([])
        print("self.total_alice_sent_int: " + str(self.total_alice_sent_int))
        print("self.total_bob_rec_int: " + str(self.total_bob_rec_int))

        #iterate through list of integers sent & received
        for index in range(len(self.total_alice_sent_int)):
            #convert each int to binlist
            bits_sent = int_to_binlist(self.total_alice_sent_int[index], 8)
            bits_rec = int_to_binlist(self.total_bob_rec_int[index], 8)

            #compare sent & rec, count how many bits flipped per byte
            bits_flipped = count_bits_flipped(bits_sent, bits_rec)

            #add number to list self.bits_flipped_by_byte
            self.bits_flipped_by_byte = np.append(self.bits_flipped_by_byte, bits_flipped)



    def graph_averages_overtime(self):
        averages_transposed = np.array(self.historical_average_rewards).T.tolist()
        x_list = range(len(averages_transposed[0]))

        figure = plt.figure(1)

        plt.title("Average Rewards vs Trials Run", fontsize=18)
        plt.xlabel('Trials Run')
        plt.ylabel('Average Rewards (Bits Flipped per Byte)')
        plt.grid(True)
        #plt.savefig('./plots/avg-rewards_vs_trials.png', format='png', dpi=300)

        color_codings = ["r-","y-","k-","b-","g-"]

        for index, avgs in enumerate(averages_transposed):
            plt.plot(x_list, avgs, color_codings[index], label=(str(self.eve_noise_arms[index])+ " dB"))

        plt.legend(loc=2)

        figure.savefig('./plots/power_penalty/avg-rewards_vs_trials.png', format='png', dpi=300)
        #plt.show()


    def graph_pickaction_choices_overtime(self):
        pickaction_transposed = np.array(self.historical_pickaction_choices).T.tolist()
        x_list = range(len(pickaction_transposed[0]))

        figure = plt.figure(2)

        title_str = "Frequency of Each Arm Chosen vs Trials Run (epsilon=" + "{0:.2f}".format(self.epsilon) + ")"
        plt.title(title_str, fontsize=18)
        plt.xlabel('Trials Run')
        plt.ylabel('Frequency Arm is Chosen')
        plt.grid(True)
        
        #plt.savefig('./plots/avg-rewards_vs_trials.png', format='png', dpi=300)

        color_codings = ["r-","y-","k-","b-","g-"]

        for index, percentage in enumerate(pickaction_transposed):
            plt.plot(x_list, percentage, color_codings[index], label=(str(self.eve_noise_arms[index])+ " dB"))

        plt.legend(loc=2)

        figure.savefig('./plots/power_penalty/pickaction_vs_trials.png', format='png', dpi=300)
        #plt.show()


def ints_to_list_of_binlists(integer_list):
    list_of_binlists = [[] for x in range(len(integer_list))]
    for index, num in enumerate(integer_list):
        num_in_bin = self.int_to_binlist(num, 8)
        list_of_binlists[index] = num_in_bin
    return(list_of_binlists)


# function to convert an integer to a list of binary numbers
def int_to_binlist(number, num_bin_digits):
    tmp_num = number
    num_in_bin = [0 for x in range(num_bin_digits)]
    for index in range(len(num_in_bin)):
        if tmp_num >= 2**(len(num_in_bin)-1-index):
            tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
            num_in_bin[index] = 1
    # print(num_in_bin)
    return num_in_bin

# counts number of bits flipped
def count_bits_flipped(sent_data, rec_data):
    num_bits_flipped = 0

    for i in range(len(sent_data)):
        if(sent_data[i] != rec_data[i]):
            num_bits_flipped = num_bits_flipped+1
    return num_bits_flipped



if __name__ == '__main__':


    """
    test example
        -4 bytes for packet
        -2 choices for jamming power: -10db or 10db
        -reward system: 
            -20 pts for important packet, 5 pts for non-important packet
            -"successful jam" means flipping at least 2 bits per byte
        -power penatly: -10 for for high power, 0 for low power

    points per packet: 20, 5, 5, 20


    """


    arms = np.array([[-100,100],[-100,100],[-100,100],[-100,100]])
    arm_counts = np.zeros((4,2)) #number of times each arm has been chosen
    arm_rewards = np.zeros((4,2)) #total reward each arm has received
    arm_average_rewards = np.zeros((4,2)) #average reward the arm receives
    artificial_rewards = np.array([20, 5, 5, 20])
    artificial_best_pattern = np.array([1,0,0,1])
    power_penalty_list = np.array([0,-10])

    num_episodes = 25

    #def __init__(self, arms, arm_counts, arm_rewards, arm_average_rewards, num_episodes, artificial_rewards, artificial_best_pattern, power_penalty_list):
    model = monte_carlo_model(arms, arm_counts, arm_rewards, arm_average_rewards, num_episodes, artificial_rewards, artificial_best_pattern, power_penalty_list)
    model.train_model(num_episodes)




"""
TO DO:
    -use vector source instead of n-timestep instantiations of topblock
    -get traffic generator working, and put model in between
    -add epsilon component (currently 100% explore)

"""