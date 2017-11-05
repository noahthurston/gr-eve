#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Eve
# Author: Noah Thurston
# Generated: Sat Nov  4 22:29:47 2017
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
from PyQt4.QtCore import QObject, pyqtSlot
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from gnuradio.qtgui import Range, RangeWidget
from optparse import OptionParser
import classify
import numpy
import numpy as np
import sip
import sys
from gnuradio import qtgui


class eve(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Eve")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Eve")
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

        self.settings = Qt.QSettings("GNU Radio", "eve")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################

        self.const_qpsk = const_qpsk = digital.constellation_qpsk().base()


        self.const_bpsk = const_bpsk = digital.constellation_bpsk().base()


        self.const_8psk = const_8psk = digital.constellation_8psk().base()


        self.const_16qam = const_16qam = digital.constellation_16qam().base()

        self.snr_db_ae = snr_db_ae = 10
        self.snr_db_ab = snr_db_ab = 10
        self.signal_len = signal_len = 1024
        self.samp_rate = samp_rate = 100000
        self.constellations = constellations = [const_bpsk, const_qpsk, const_8psk, const_16qam]
        self.const_index = const_index = 3
        self.ab_mute = ab_mute = False

        ##################################################
        # Blocks
        ##################################################
        self._snr_db_ae_range = Range(-10, 15, 1, 10, 200)
        self._snr_db_ae_win = RangeWidget(self._snr_db_ae_range, self.set_snr_db_ae, "snr_db_ae", "counter_slider", float)
        self.top_layout.addWidget(self._snr_db_ae_win)
        self._snr_db_ab_range = Range(-10, 15, 1, 10, 200)
        self._snr_db_ab_win = RangeWidget(self._snr_db_ab_range, self.set_snr_db_ab, "snr_db_ab", "counter_slider", float)
        self.top_layout.addWidget(self._snr_db_ab_win)
        self._const_index_options = (0, 1, 2, 3, )
        self._const_index_labels = ('BPSK', 'QPSK', '8PSK', '16QAM', )
        self._const_index_group_box = Qt.QGroupBox('Constellation')
        self._const_index_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._const_index_button_group = variable_chooser_button_group()
        self._const_index_group_box.setLayout(self._const_index_box)
        for i, label in enumerate(self._const_index_labels):
        	radio_button = Qt.QRadioButton(label)
        	self._const_index_box.addWidget(radio_button)
        	self._const_index_button_group.addButton(radio_button, i)
        self._const_index_callback = lambda i: Qt.QMetaObject.invokeMethod(self._const_index_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._const_index_options.index(i)))
        self._const_index_callback(self.const_index)
        self._const_index_button_group.buttonClicked[int].connect(
        	lambda i: self.set_const_index(self._const_index_options[i]))
        self.top_layout.addWidget(self._const_index_group_box)
        self._ab_mute_options = (False, True, )
        self._ab_mute_labels = ("Don't Block", 'Block', )
        self._ab_mute_group_box = Qt.QGroupBox('Mute AB Output')
        self._ab_mute_box = Qt.QVBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._ab_mute_button_group = variable_chooser_button_group()
        self._ab_mute_group_box.setLayout(self._ab_mute_box)
        for i, label in enumerate(self._ab_mute_labels):
        	radio_button = Qt.QRadioButton(label)
        	self._ab_mute_box.addWidget(radio_button)
        	self._ab_mute_button_group.addButton(radio_button, i)
        self._ab_mute_callback = lambda i: Qt.QMetaObject.invokeMethod(self._ab_mute_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._ab_mute_options.index(i)))
        self._ab_mute_callback(self.ab_mute)
        self._ab_mute_button_group.buttonClicked[int].connect(
        	lambda i: self.set_ab_mute(self._ab_mute_options[i]))
        self.top_layout.addWidget(self._ab_mute_group_box)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
        	1024, #size
        	"", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
        	1024, #size
        	"", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
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
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=numpy.sqrt(10.0**(-snr_db_ab/10.0)/2),
        	frequency_offset=0.0,
        	epsilon=1.0,
        	taps=(1.0, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, int(np.log2(constellations[const_index].arity())), "", False, gr.GR_MSB_FIRST)
        self.blocks_mute_xx_0 = blocks.mute_cc(bool(ab_mute))
        self.blocks_message_debug_0 = blocks.message_debug()
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 10000)), True)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.classify_trained_model_classifier_vc_0, 'classification_info'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_mute_xx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.classify_trained_model_classifier_vc_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_mute_xx_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.channels_channel_model_0_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "eve")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

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

    def get_snr_db_ab(self):
        return self.snr_db_ab

    def set_snr_db_ab(self, snr_db_ab):
        self.snr_db_ab = snr_db_ab
        self.channels_channel_model_0.set_noise_voltage(numpy.sqrt(10.0**(-self.snr_db_ab/10.0)/2))

    def get_signal_len(self):
        return self.signal_len

    def set_signal_len(self, signal_len):
        self.signal_len = signal_len

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

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
        self._const_index_callback(self.const_index)
        self.digital_chunks_to_symbols_xx_0.set_symbol_table((self.constellations[self.const_index].points()))
        self.blocks_repack_bits_bb_0.set_k_and_l(8,int(np.log2(self.constellations[self.const_index].arity())))

    def get_ab_mute(self):
        return self.ab_mute

    def set_ab_mute(self, ab_mute):
        self.ab_mute = ab_mute
        self._ab_mute_callback(self.ab_mute)
        self.blocks_mute_xx_0.set_mute(bool(self.ab_mute))


def main(top_block_cls=eve, options=None):

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


if __name__ == '__main__':
    main()
