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

    def __init__(self, eve_noise_db=1, channel_noise_db=1, max_items=8):
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
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, np.random.randint(0, 2, 1000000)), True)
        self.analog_fastnoise_source_x_0_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(self.eve_noise_db/20.0), 0, 2**16)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(self.channel_noise_db/20.0), 0, 2**16)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.analog_fastnoise_source_x_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_1, 0))
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




#h5 table for keeping data for reinforcement learning
class LearningTable(IsDescription):
    packet_time_ID = Int64Col()
    eve_noise_db = Float64Col() #this is the action that must be picked by the model
    reward = Float64Col()
    
    #eve_noise_db = bob snr

    #modulation is the state for now
    modulation_detected = StringCol(16) #max 16 chars
    #power_detected = Float64Col()


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
class eve_learning_model():
    def __init__(self, epsilon, eve_noise_patterns, arm_counts, average_rewards, bytes_per_packet, with_power_penalty, bits_flipped_threshold):
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

    # main function, iterates through trials 
    def train_model(self, num_trials):
        print("beginning training")

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

    # function instantiates top block and runs single step
    def run_trial(self):

        #picks action to be taken during this packet
        chosen_arm = self.pick_arm()
        chosen_eve_noise_db = self.eve_noise_arms[chosen_arm]

        #create topblock with parameters chosen by pick_action and run
        #__init__(self, eve_noise_db=1, channel_noise_db=1, max_items=100):
        print("bytes: %d" % self.bytes_per_packet)
        tb = eve_re_learn_testbed_graph(eve_noise_db=chosen_eve_noise_db, channel_noise_db=-100, max_items=self.bytes_per_packet)
        tb.start()
        tb.wait()

        #fetch data
        alice_sent = tb.blocks_vector_sink_alice.data()
        bob_rec = tb.blocks_vector_sink_bob.data() 

        #calculate reward based on results
        reward = self.calculate_reward(alice_sent, bob_rec, chosen_arm)

        self.update_model(chosen_arm, reward)




    # given the current state and inputs, it calculates the action to be taken
    def pick_arm(self):

        if random.random() > self.epsilon:
            chosen_arm = self.average_rewards.index(max(self.average_rewards))
        else:
            chosen_arm = random.randrange(len(self.eve_noise_arms))

        print("chosen_arm:%d , eve noise db:%d" % (chosen_arm, self.eve_noise_arms[chosen_arm]))
        return chosen_arm

        ### need some reinforcement learning code here to decide the next action
        ### will need access to the table of past actions and current state 
        #eve_noise_db = 0
        #print("picking action, eve_noise_db: %d" % eve_noise_db)
        #return(eve_noise_db)

    # given what alice sent and bob received, it scores Eve's decision
    def calculate_reward(self, alice_sent, bob_rec, chosen_arm):
        alice_sent_bin_lists = self.ints_to_list_of_binlists(alice_sent)
        bob_rec_bin_lists = self.ints_to_list_of_binlists(bob_rec)
        bits_flipped = self.count_bits_flipped(alice_sent_bin_lists, bob_rec_bin_lists)

        if self.with_power_penalty == True:
            if bits_flipped >= self.bits_flipped_threshold*self.bytes_per_packet:
                reward = 5 - (self.eve_noise_arms[chosen_arm]+10)*0.1
            else:
                reward = 0
        else:
            reward = bits_flipped / self.bytes_per_packet

        print("reward: %d" %reward)
        return(reward)

    ### needs works
    def update_model(self, chosen_arm, reward):
        self.arm_counts[chosen_arm] = self.arm_counts[chosen_arm]+1
        n = float(self.arm_counts[chosen_arm])

        print(average_rewards)

        avg_reward = self.average_rewards[chosen_arm]
        new_avg_reward = ((n-1)/n)*avg_reward + reward/n
        self.average_rewards[chosen_arm] = new_avg_reward


    def ints_to_list_of_binlists(self, integer_list):
        list_of_binlists = [[] for x in range(len(integer_list))]
        for index, num in enumerate(integer_list):
            num_in_bin = self.int_to_binlist(num, 8)
            list_of_binlists[index] = num_in_bin
        return(list_of_binlists)


    # function to convert an integer to a list of binary numbers
    def int_to_binlist(self, number, num_bin_digits):
        tmp_num = number
        num_in_bin = [0 for x in range(num_bin_digits)]
        for index in range(len(num_in_bin)):
            if tmp_num >= 2**(len(num_in_bin)-1-index):
                tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
                num_in_bin[index] = 1
        # print(num_in_bin)
        return num_in_bin

    # counts number of bits flipped
    def count_bits_flipped(self, sent_data, rec_data):
        num_bits_flipped = 0
        #print(sent_data)
        #print(rec_data)

        for i in range(len(sent_data)):
            for j in range(len(sent_data[i])):
                if(sent_data[i][j] != rec_data[i][j]):
                    num_bits_flipped = num_bits_flipped+1
        return num_bits_flipped

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




if __name__ == '__main__':

    eve_noise_arms = range(-10,11,5)
    arm_counts = [0 for x in range(len(eve_noise_arms))]
    average_rewards = [0 for x in range(len(eve_noise_arms))]
    print("eve_noise_arms: " + str(eve_noise_arms))
    print("arm_counts: " + str(arm_counts))
    print("average_rewards: " + str(average_rewards))


    model = eve_learning_model(0.40, eve_noise_arms, arm_counts, average_rewards, 8, True, 2)
    model.train_model(200)

