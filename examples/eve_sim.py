#!/usr/bin/env python
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve Sim
# Author: Noah Thurston
# Generated: Sun Oct 29 09:50:29 2017
#
# Script to run simulation trials of eve_sim_flowgraph
# current script in gr-eve_mod/examples
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

from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import cognition
import classify
import numpy
import numpy as np
import pmt
import sip
import sys
import csv
import matplotlib.pyplot as plt


class eve_sim(gr.top_block):

    def __init__(self, snr_db_ae = 15, signal_len = 1024, samp_rate = 100000, samples = 1000, const_index = 3):
        gr.top_block.__init__(self, "Eve Sim")




        ##################################################
        # Variables
        ##################################################

        self.const_qpsk = const_qpsk = digital.constellation_qpsk().base()
        self.const_bpsk = const_bpsk = digital.constellation_bpsk().base()
        self.const_8psk = const_8psk = digital.constellation_8psk().base()
        self.const_16qam = const_16qam = digital.constellation_16qam().base()

        self.snr_db_ae = snr_db_ae # = 15
        self.signal_len = signal_len # = 1024
        self.samp_rate = samp_rate # = 100000
        self.constellations = constellations = [const_bpsk, const_qpsk, const_8psk, const_16qam]
        self.const_index = const_index

        ##################################################
        # Blocks
        ##################################################
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc((constellations[const_index].points()), 1)
        self.classify_trained_model_classifier_vc_0 = classify.trained_model_classifier_vc(64, '/home/gvanhoy/gr-classify/apps/cumulant_classifier.pkl')
        self.channels_channel_model_0_0 = channels.channel_model(
        	noise_voltage=numpy.sqrt(10.0**(-snr_db_ae/10.0)/2),
        	frequency_offset=0.0,
        	epsilon=1.0,
        	taps=(1.0, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, int(np.log2(constellations[const_index].arity())), "", False, gr.GR_MSB_FIRST)
        self.blocks_head_1 = blocks.head(gr.sizeof_gr_complex*64, samples)

        #message block to pull messages out of
        self.blocks_message_debug_0 = blocks.message_debug()
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 10000)), True)

        ##################################################
        # Connections
        ##################################################
        #self.msg_connect((self.classify_trained_model_classifier_vc_0, 'classification_info'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.classify_trained_model_classifier_vc_0, 'classification_info'), (self.blocks_message_debug_0, 'store'))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        
        #self.connect((self.blocks_stream_to_vector_0, 0), (self.classify_trained_model_classifier_vc_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_head_1, 0))
        self.connect((self.blocks_head_1, 0), (self.classify_trained_model_classifier_vc_0, 0))

        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.channels_channel_model_0_0, 0))

        

    def get_const_qpsk(self):
        return self.const_qpsk

    def set_const_qpsk(self, const_qpsk):
        self.const_qpsk = const_qpsk
        self.set_constellations([self.const_bpsk, self.const_qpsk, self.const_8psk, self.const_16qam])

    def get_const_bpsk(self):
        return self.const_bpsk

    def set_const_bpsk(self, const_bpsk):
        self.const_bpsk = const_bpsk
        self.set_constellations([self.const_bpsk, self.const_qpsk, self.const_8psk, self.const_16qam])

    def get_const_8psk(self):
        return self.const_8psk

    def set_const_8psk(self, const_8psk):
        self.const_8psk = const_8psk
        self.set_constellations([self.const_bpsk, self.const_qpsk, self.const_8psk, self.const_16qam])

    def get_const_16qam(self):
        return self.const_16qam

    def set_const_16qam(self, const_16qam):
        self.const_16qam = const_16qam
        self.set_constellations([self.const_bpsk, self.const_qpsk, self.const_8psk, self.const_16qam])

    def get_snr_db_ae(self):
        return self.snr_db_ae

    def set_snr_db_ae(self, snr_db_ae):
        self.snr_db_ae = snr_db_ae
        self.channels_channel_model_0_0.set_noise_voltage(10.0**(-self.snr_db_ae/20.0))

    def get_signal_len(self):
        return self.signal_len

    def set_signal_len(self, signal_len):
        self.signal_len = signal_len

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)

    def get_constellations(self):
        return self.constellations

    def set_constellations(self, constellations):
        self.constellations = constellations
        self.digital_chunks_to_symbols_xx_0.set_symbol_table((self.constellations[self.const_index].points()))
        self.blocks_repack_bits_bb_0.set_k_and_l(8,int(np.log2(self.constellations[self.const_index].arity())))

    def get_const_index(self):
        return self.const_index

    def set_const_index(self, const_index):
        self.const_index = const_index
        self.digital_chunks_to_symbols_xx_0.set_symbol_table((self.constellations[self.const_index].points()))
        self.blocks_repack_bits_bb_0.set_k_and_l(8,int(np.log2(self.constellations[self.const_index].arity())))


def main(top_block_cls=eve_sim, options=None):

    tb = top_block_cls()
    tb.start()

    tb.wait()
    
    for i in range(10):
        print pmt.symbol_to_string(pmt.cdr((tb.blocks_message_debug_0.get_message(i))))
        print tb.blocks_message_debug_0.num_messages()

    tb.stop()
    tb.wait()


def run_sims(top_block_cls=eve_sim, options=None):
    #dictionary of modulations and indexes
    mod  = {'BPSK':0, 'QPSK':1, '8PSK':2, '16QAM':3}

    #paramters when running a sim, could be organized into a class for a sim
    snr_db_ae = 10;
    signal_len = 1024;
    samp_rate = 100000;
    samples_to_check = 100;
    samples = 10000;

    #how to run for only a couple seconds or get only a limited number of samples?

    #2d lists to hold sim results (copies for results as nominal counts and percents)
    sim_results = [[0 for col in range(5)] for row in range(4)]
    sim_results_percent = [[0 for col in range(5)] for row in range(4)]


    #sims is a list of topblocks
    sims = [];
    curr_message = ''

    #iterate through the 4 possible modulations
    for mod_index in range(4):
        sims.append(top_block_cls(snr_db_ae, signal_len, samp_rate, samples, mod_index))
        sims[mod_index].start()
        sims[mod_index].wait()
        sims[mod_index].stop()
        print mod_index
        print 'Number of messages received:',sims[mod_index].blocks_message_debug_0.num_messages()
        
        #tally recieved symbols and how they were classified
        for curr_samp in range(samples):
            curr_message = pmt.symbol_to_string(pmt.cdr(sims[mod_index].blocks_message_debug_0.get_message(curr_samp)))
            sim_results[mod_index][mod[curr_message]] +=1
            sim_results[mod_index][4] +=1

    #print output of the sim 
    print 'BPSK:',sim_results[mod['BPSK']]
    print 'QPSK:',sim_results[mod['QPSK']]
    print '8PSK:',sim_results[mod['8PSK']]
    print '16QAM:',sim_results[mod['16QAM']]

    #calculate the confusion matrix on a percent basis
    for curr_row in range(4):
        for curr_col in range(4):
            sim_results_percent[curr_row][curr_col] = float(sim_results[curr_row][curr_col]) / float(sim_results[curr_row][4])
        sim_results_percent[curr_row][4] = 1




    #write results to a file as a csv
    output_file = open("confusion_matrix.csv", 'w')
    write_csv(sim_results, output_file)
    write_csv(sim_results_percent, output_file)
    output_file.close()




def write_csv(sim_results, output_file):
    
    csv_column_labels = ["BPSK_classified",'QPSK_classified','8PSK_classified','16QAM_classified','total_classified']

    csv_writer = csv.writer(output_file)
    csv_writer.writerow(csv_column_labels)

    csv_writer.writerow(["BPSK_sim:"] + sim_results[0])
    csv_writer.writerow(["QPSK_sim:"] + sim_results[1])
    csv_writer.writerow(["8PSK_sim:"] + sim_results[2])
    csv_writer.writerow(["16QAM_sim:"] + sim_results[3])

    output_file.write('\n')



if __name__ == '__main__':
    run_sims()
