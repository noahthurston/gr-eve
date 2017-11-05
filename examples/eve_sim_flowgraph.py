#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve Sim Flowgraph
# Author: Noah Thurston
# Generated: Wed Nov  1 23:01:28 2017
##################################################

from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import classify
import numpy
import numpy as np


class eve_sim_flowgraph(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Eve Sim Flowgraph")

        ##################################################
        # Variables
        ##################################################

        self.const_qpsk = const_qpsk = digital.constellation_qpsk().base()


        self.const_bpsk = const_bpsk = digital.constellation_bpsk().base()


        self.const_8psk = const_8psk = digital.constellation_8psk().base()


        self.const_16qam = const_16qam = digital.constellation_16qam().base()

        self.snr_db_ae = snr_db_ae = 15
        self.signal_len = signal_len = 1024
        self.samples = samples = 10000
        self.samp_rate = samp_rate = 100000
        self.constellations = constellations = [const_bpsk, const_qpsk, const_8psk, const_16qam]
        self.const_index = const_index = 2

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
        self.blocks_message_debug_0 = blocks.message_debug()
        self.blocks_head_1 = blocks.head(gr.sizeof_gr_complex*64, samples)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 10000)), True)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.classify_trained_model_classifier_vc_0, 'classification_info'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.classify_trained_model_classifier_vc_0, 'classification_info'), (self.blocks_message_debug_0, 'store'))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_head_1, 0), (self.classify_trained_model_classifier_vc_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_head_1, 0))
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
        self.channels_channel_model_0_0.set_noise_voltage(numpy.sqrt(10.0**(-self.snr_db_ae/10.0)/2))

    def get_signal_len(self):
        return self.signal_len

    def set_signal_len(self, signal_len):
        self.signal_len = signal_len

    def get_samples(self):
        return self.samples

    def set_samples(self, samples):
        self.samples = samples
        self.blocks_head_1.set_length(self.samples)

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


def main(top_block_cls=eve_sim_flowgraph, options=None):

    tb = top_block_cls()
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
