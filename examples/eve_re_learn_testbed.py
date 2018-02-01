#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve Re Learn Testbed Graph
# Generated: Tue Jan 23 11:09:53 2018
##################################################

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

class eve_re_learn_testbed_graph(gr.top_block, Qt.QWidget):

    def __init__(self, eve_noise_db=1, channel_noise_db=1, max_items=100):
        gr.top_block.__init__(self, "Eve Re Learn Testbed Graph")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Eve Re Learn Testbed Graph")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "eve_re_learn_testbed_graph")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 0
        self.samp_rate = samp_rate = 1000000
        self.max_items = max_items = 100

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

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

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


#table to hold data from a single packet, MAY NOT BE NEEDED
class PacketTable(IsDescription):
    time_ID = Int64Col()
    alice_sent = Int8Col()
    bob_rec = Int8Col()
    calc_bits_flipped = Int8Col()



#created separate class so that the reinforcement learning table could be accessed easily anywhere within the function
class eve_learning_model():
    def __init__(self, reinforcement_learning_table, max_items=100):
        #epsilon-Greedy parameters
        #self.epsilon = epsilon
        #self.arms = arms #list of values for 


        self.max_items = max_items
        self.reinforcement_learning_table = reinforcement_learning_table
        self.current_channel_noise_db = 0
        self.current_time = 0
        ### add any other model parameters

    # main function, iterates through trials 
    def train_model(self, num_trials):
        print("beginning training")

        # loop through trainning trialss
        for trial in range(num_trials):
            print("\nrunning trial #%d" %trial)
            self.run_trial() 

    # function instantiates top block and runs single step
    def run_trial(self):

        #picks action to be taken during this packet
        eve_noise_db = self.pick_action()

        #create topblock with parameters chosen by pick_action and run
        #__init__(self, eve_noise_db=1, channel_noise_db=1, max_items=100):
        tb = eve_re_learn_testbed_graph(eve_noise_db=eve_noise_db, channel_noise_db=self.current_channel_noise_db, max_items=self.max_items)
        tb.start()
        tb.wait()

        #fetch data
        alice_sent = tb.blocks_vector_sink_alice.data()
        bob_rec = tb.blocks_vector_sink_bob.data() 

        #calculate reward based on results
        reward = self.calculate_reward(eve_noise_db, alice_sent, bob_rec)

        #put results in table
        new_row = self.reinforcement_learning_table.row
        self.current_time = self.current_time+1
        new_row['packet_time_ID'] = self.current_time
        new_row['eve_noise_db'] = eve_noise_db
        new_row['reward'] = reward
        new_row.append()

    # given what alice sent and bob received, it scores Eve's decision
    def calculate_reward(self, eve_noise_db, alice_sent, bob_rec):

        ### need some reinforcement learning code here to calculate appropriate reward for actions
        ### will need to know actions it took and results
        reward = 0
        print("reward: %d" %reward)
        return(reward)

    # given the current state and inputs, it calculates the action to be taken
    def pick_action(self):

        ### need some reinforcement learning code here to decide the next action
        ### will need access to the table of past actions and current state 
        eve_noise_db = 0
        print("picking action, eve_noise_db: %d" % eve_noise_db)
        return(eve_noise_db)


    #### these functions may not be needed

    # data handler for storing a single 8-bit section of a packet
    def eightbit_data_handler(self, alice_sent, bob_rec, table):
        # put data into group.table
        myRow = table.row
        for i in range(len(alice_sent)):
            myRow['time_ID'] = i
            myRow['alice_sent'] = alice_sent[i]
            myRow['bob_rec'] = bob_rec[i]
            myRow['calc_bits_flipped'] = -1
            myRow.append()

        #flushes table IO buffer
        table.flush()    

    # function to convert an integer to a list of binary numbers
    def int_to_binlist(self, num_int, num_bin_digits):
        tmp_num = num_int
        num_in_bin = np.zeros(num_bin_digits)
        for index in range(len(num_in_bin)):
            if tmp_num >= 2**(len(num_in_bin)-1-index):
                tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
                num_in_bin[index] = 1
        # print(num_in_bin)
        return num_in_bin

    # counts number of bits flipped
    def count_bits_flipped(self, bin_list1, bin_list2):
        num_bits_flipped = 0
        for index in range(len(bin_list1)):
            if bin_list1[index] != bin_list2[index]:
                num_bits_flipped += 1
        return num_bits_flipped




##################################################################################################################

def test_bits_flipped_vs_noise(top_block_cls=eve_re_learn_testbed_graph, options=None):
    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    db_list = range(-20,21)
    avg_bits_flipped_list = []

    for test_db in db_list:

        tb = top_block_cls(test_db, -100)
        tb.start()
        tb.wait()

        # grab data from sink
        alice_sent = tb.blocks_vector_sink_alice.data()
        bob_rec = tb.blocks_vector_sink_bob.data()

        # create np array to hold sent and rec data
        data_sent_n_rec = np.zeros((tb.max_items,2))
        #print(data_sent_n_rec)

        for index, val in enumerate(alice_sent):
            data_sent_n_rec[index][0] = val

        for index, val in enumerate(bob_rec):
            data_sent_n_rec[index][1] = val

        #print(data_sent_n_rec)
        #print("\n\n")

        bits_flipped_list = []

        for row in data_sent_n_rec:
            alice_sent_bin_list = int_to_binlist(int(row[0]), 8)
            bob_rec_bin_list = int_to_binlist(int(row[1]), 8)
            num_bits_flipped = count_bits_flipped(alice_sent_bin_list, bob_rec_bin_list)

            bits_flipped_list.append(num_bits_flipped)

            # print(bin(int(row[0])))
            # print("%s" % (bin(int(row[0]))))
            # print("%s\t%d bits flipped" % (bin(int(row[1])),num_bits_flipped))
            # print("-----------")

        avg_bits_flipped = sum(bits_flipped_list)/float(len(bits_flipped_list))
        avg_bits_flipped_list.append(avg_bits_flipped)
        print("%d db, avg bits flipped: %f" % (test_db, avg_bits_flipped))

    # def int_to_binlist(num_int, num_bin_digits):
    # def count_bits_flipped(bin_list1, bin_list2):

    plt.plot(db_list, avg_bits_flipped_list, 'r-')
    plt.title("Bits Flipped vs Eve's Generated Noise (db)", fontsize=18)
    plt.xlabel("Eve's Generated Noise (db)")
    plt.ylabel('Bits Flipped')
    plt.grid(True)
    plt.savefig('./plots/bits-flipped_vs_eve-noise_large_1.png', format='png', dpi=300)

    tb.show()


    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()

# function to convert an integer to a list of binary numbers
def int_to_binlist(num_int, num_bin_digits):
    tmp_num = num_int
    num_in_bin = np.zeros(num_bin_digits)
    for index in range(len(num_in_bin)):
        if tmp_num >= 2**(len(num_in_bin)-1-index):
            tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
            num_in_bin[index] = 1
    # print(num_in_bin)
    return num_in_bin

# counts number of bits flipped
def count_bits_flipped(bin_list1, bin_list2):
    num_bits_flipped = 0
    for index in range(len(bin_list1)):
        if bin_list1[index] != bin_list2[index]:
            num_bits_flipped += 1
    return num_bits_flipped
###################################################################################################################



if __name__ == '__main__':

    #boiler plate QT GUI code
    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)
    
    """
    h5file = open_file("./test/learning_model_table.h5", mode="w", title="Test Table Title`")
    group = h5file.create_group("/", 'sim_group', 'Group information')
    table = h5file.create_table(group, 'test_nodename', LearningTable, "The Best Table Title")
    
    model = eve_learning_model(table)
    model.train_model(3)

    h5file.close()
    """

    test_bits_flipped_vs_noise()






### test functions, keep for example code

def test_h5_table(top_block_cls=eve_re_learn_testbed_graph, options=None):
    # start app, delete QT GUI later
    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    # create block
    # run block to max
    test_db = 0
    tb = top_block_cls(test_db,test_db)
    tb.start()
    tb.wait()

    # create table
    h5file = open_file("./tables/table_" + str(test_db) + "db.h5", mode="w", title="Test Table Title`")
    group = h5file.create_group("/", 'sim_group', 'Group information')
    table = h5file.create_table(group, 'test_nodename', PacketTable, "The Best Table Title")

    # grab data
    alice_sent = tb.blocks_vector_sink_alice.data()
    bob_rec = tb.blocks_vector_sink_bob.data()

    # send data to data handler
        # put data into table
    eightbit_data_handler(alice_sent, bob_rec, table)

    #### test debug
    print("\nSTART READING")
    table = h5file.root.sim_group.test_nodename
    all_IDs = [x['time_ID'] for x in table.iterrows()]
    all_alice_sent = [x['alice_sent'] for x in table.iterrows()]
    all_bob_rec = [x['bob_rec'] for x in table.iterrows()]

    for i in range(len(all_IDs)):
        print("ID:% d\tAlice Sent: %d\t\tBob Rec: %d" % (all_IDs[i], all_alice_sent[i], all_bob_rec[i]))

    h5file.close()

def test_bits_flipped_vs_noise(top_block_cls=eve_re_learn_testbed_graph, options=None):
    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    db_list = range(-20,21)
    avg_bits_flipped_list = []

    for test_db in db_list:

        tb = top_block_cls(test_db,test_db)
        tb.start()
        tb.wait()

        # grab data from sink
        alice_sent = tb.blocks_vector_sink_alice.data()
        bob_rec = tb.blocks_vector_sink_bob.data()

        # create np array to hold sent and rec data
        data_sent_n_rec = np.zeros((tb.max_items,2))
        #print(data_sent_n_rec)

        for index, val in enumerate(alice_sent):
            data_sent_n_rec[index][0] = val

        for index, val in enumerate(bob_rec):
            data_sent_n_rec[index][1] = val

        #print(data_sent_n_rec)
        #print("\n\n")

        bits_flipped_list = []

        for row in data_sent_n_rec:
            alice_sent_bin_list = int_to_binlist(int(row[0]), 8)
            bob_rec_bin_list = int_to_binlist(int(row[1]), 8)
            num_bits_flipped = count_bits_flipped(alice_sent_bin_list, bob_rec_bin_list)

            bits_flipped_list.append(num_bits_flipped)

            # print(bin(int(row[0])))
            # print("%s" % (bin(int(row[0]))))
            # print("%s\t%d bits flipped" % (bin(int(row[1])),num_bits_flipped))
            # print("-----------")

        avg_bits_flipped = sum(bits_flipped_list)/float(len(bits_flipped_list))
        avg_bits_flipped_list.append(avg_bits_flipped)
        print("%d db, avg bits flipped: %f" % (test_db, avg_bits_flipped))

    # def int_to_binlist(num_int, num_bin_digits):
    # def count_bits_flipped(bin_list1, bin_list2):

    plt.plot(db_list, avg_bits_flipped_list, 'r-')
    plt.title("Bits Flipped vs Eve and Channel Noise", fontsize=18)
    plt.xlabel('Eve and Channel Noise (db)')
    plt.ylabel('Bits Flipped')
    plt.grid(True)
    plt.savefig('./plots/bits-flipped_vs_channel-noise_large_1.png', format='png', dpi=300)

    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()

# function to convert an integer to a list of binary numbers
def int_to_binlist(num_int, num_bin_digits):
    tmp_num = num_int
    num_in_bin = np.zeros(num_bin_digits)
    for index in range(len(num_in_bin)):
        if tmp_num >= 2**(len(num_in_bin)-1-index):
            tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
            num_in_bin[index] = 1
    # print(num_in_bin)
    return num_in_bin

# counts number of bits flipped
def count_bits_flipped(bin_list1, bin_list2):
    num_bits_flipped = 0
    for index in range(len(bin_list1)):
        if bin_list1[index] != bin_list2[index]:
            num_bits_flipped += 1
    return num_bits_flipped



#run test with channel noise being -100 (basically nothing), so it actually is eve channel noise, not bob SNR