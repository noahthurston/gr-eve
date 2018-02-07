#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve Re Learn Testbed Noqt
# Generated: Tue Feb  6 22:26:35 2018
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy


class eve_re_learn_testbed_noqt(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Eve Re Learn Testbed Noqt")

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 0
        self.samp_rate = samp_rate = 32000
        self.max_items = max_items = 80

        self.const = const = digital.constellation_calcdist(([-1-1j, -1+1j, 1+1j, 1-1j]), ([0, 1, 3, 2]), 4, 1).base()


        ##################################################
        # Blocks
        ##################################################
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(const)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc((const.points()), 1)
        self.blocks_vector_sink_x_1 = blocks.vector_sink_b(1)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_b(1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_repack_bits_bb_1 = blocks.repack_bits_bb(1, 8, "", False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(1, const.bits_per_symbol(), "", False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(const.bits_per_symbol(), 8, "", False, gr.GR_LSB_FIRST)
        self.blocks_head_0_0 = blocks.head(gr.sizeof_char*1, max_items)
        self.blocks_head_0 = blocks.head(gr.sizeof_char*1, max_items)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 2, 1000000)), True)
        self.analog_fastnoise_source_x_0_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0, 2**16)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0, 2**16)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.analog_fastnoise_source_x_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_1, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_vector_sink_x_0, 0))
        self.connect((self.blocks_head_0_0, 0), (self.blocks_vector_sink_x_1, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_repack_bits_bb_1, 0), (self.blocks_head_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.blocks_repack_bits_bb_0, 0))

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


def main(top_block_cls=eve_re_learn_testbed_noqt, options=None):

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
