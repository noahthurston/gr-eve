from matplotlib import pyplot as plt
import numpy as np
import time

import csv

class Plot_Datarate_vs_SNR:

	def __init__(self):
		#list of range values, should be same as the sim
		self.snr_range = range(-10,10,1)

		#2d list for data read from csv
		self.csv_data = []

		#calculated data throughput values, each of the 4 lists should be plotted
		self.data_throughput = []

		self.load_data_from_csv()

		self.crunch_plot_datathroughput_when_targeted()

		#self.calculate_data_throughput()
		#self.plot_data()


	def load_data_from_csv(self):
		#load saved sim data from csv
		print "loading data"

		input_csv_file = open("confusion_matrix_forplot.csv", 'r')
		csv_reader = csv.reader(input_csv_file)

		for row in csv_reader:
			self.csv_data.append(row)

		#debug
		#print self.csv_data

	def calculate_data_throughput(self):
		#using loaded data to create list of datathroughput values at each SNR and modulation
		print "calculating data throughput"

		for snr_row in range(1,len(self.csv_data),1):

			#subtract from the max possible data throughput the percent classified of each target classification and the true data value of the symbol that would've been jammed
			BPSK_data_throughput = 2.5 - 0.25*(float(self.csv_data[snr_row][0*4+ 1])*1 + float(self.csv_data[snr_row][1*4+ 1])*2 + float(self.csv_data[snr_row][2*4+ 1])*3 + float(self.csv_data[snr_row][3*4+ 1])*4)
			QPSK_data_throughput = 2.5 - 0.25*(float(self.csv_data[snr_row][0*4+ 2])*1 + float(self.csv_data[snr_row][1*4+ 2])*2 + float(self.csv_data[snr_row][2*4+ 2])*3 + float(self.csv_data[snr_row][3*4+ 2])*4)
			eightPSK_data_throughput = 2.5 - 0.25*(float(self.csv_data[snr_row][0*4+ 3])*1 + float(self.csv_data[snr_row][1*4+ 3])*2 + float(self.csv_data[snr_row][2*4+ 3])*3 + float(self.csv_data[snr_row][3*4+ 3])*4)
			sixteenQAM_data_throughput = 2.5 - 0.25*(float(self.csv_data[snr_row][0*4+ 4])*1 + float(self.csv_data[snr_row][1*4+ 4])*2 + float(self.csv_data[snr_row][2*4+ 4])*3 + float(self.csv_data[snr_row][3*4+ 4])*4)

			self.data_throughput.append([float(self.csv_data[snr_row][0]), BPSK_data_throughput, QPSK_data_throughput, eightPSK_data_throughput, sixteenQAM_data_throughput])

		#debug
		#print self.data_throughput


	def plot_data(self):
		#plot loaded sim data from csv
		print "plotting data"

		
		for mod_graph in range(1,5,1):
			x_vals = []
			y_vals = []

			for snr_row in range(len(self.data_throughput)):
				x_vals.append(self.data_throughput[snr_row][0])
				y_vals.append(self.data_throughput[snr_row][mod_graph])

			plt.figure(mod_graph)
			plt.plot(x_vals, y_vals)


		self.save_figure(1, 'BPSK Data Throughput vs SNR', 'BPSK')
		self.save_figure(2, 'QPSK Data Throughput vs SNR', 'QPSK')
		self.save_figure(3, '8PSK Data Throughput vs SNR', 'eightPSK')
		self.save_figure(4, '16QAM Data Throughput vs SNR', 'sixteenQAM')
		

	def crunch_plot_datathroughput_when_targeted(self):
		print "crunching and plotting the data throughput vs SNR when using only a single modulation and all 16QAM is jammed"
		
		#16QAM Jammed
		x_vals = []
		y_vals_BPSK = []
		y_vals_QPSK = []
		y_vals_eightPSK = []
		y_vals_sixteenQAM = []

		for snr_row in range(1,len(self.csv_data),1):
			x_vals.append(float(self.csv_data[snr_row][0]))
			y_vals_BPSK.append(1*(1-float(self.csv_data[snr_row][0*4 + 4])))
			y_vals_QPSK.append(2*(1-float(self.csv_data[snr_row][1*4 + 4])))
			y_vals_eightPSK.append(3*(1-float(self.csv_data[snr_row][2*4 + 4])))
			y_vals_sixteenQAM.append(4*(1-float(self.csv_data[snr_row][3*4 + 4])))

		plt.figure(1)
		plt.plot(x_vals, y_vals_BPSK, label='BPSK', marker='o')
		plt.plot(x_vals, y_vals_QPSK, label='QPSK', marker='o')
		plt.plot(x_vals, y_vals_eightPSK, label='8SPK', marker='o')
		plt.plot(x_vals, y_vals_sixteenQAM, label='16QAM', marker='o')

		plt.title("Data Throughput by Modulation with 16QAM Jammed", fontsize=18)
		plt.xlabel('SNR of Eve')
		plt.ylabel('AB Data Throughput')
		plt.ylim(0,4)
		plt.grid(True)
		plt.legend(loc='lower left')

		#plt.show()


		#16QAM and 8PSK Jammed
		x_vals = []
		y_vals_BPSK = []
		y_vals_QPSK = []
		y_vals_eightPSK = []
		y_vals_sixteenQAM = []

		for snr_row in range(1,len(self.csv_data),1):
			x_vals.append(float(self.csv_data[snr_row][0]))
			y_vals_BPSK.append(1*(1-float(self.csv_data[snr_row][0*4 + 4])- float(self.csv_data[snr_row][0*4 + 3])))
			y_vals_QPSK.append(2*(1-float(self.csv_data[snr_row][1*4 + 4])- float(self.csv_data[snr_row][1*4 + 3])))
			y_vals_eightPSK.append(3*(1-float(self.csv_data[snr_row][2*4 + 4])- float(self.csv_data[snr_row][2*4 + 3])))
			y_vals_sixteenQAM.append(4*(1-float(self.csv_data[snr_row][3*4 + 4])- float(self.csv_data[snr_row][3*4 + 3])))

		plt.figure(2)
		plt.plot(x_vals, y_vals_BPSK, label='BPSK', marker='o')
		plt.plot(x_vals, y_vals_QPSK, label='QPSK', marker='o')
		plt.plot(x_vals, y_vals_eightPSK, label='8SPK', marker='o')
		plt.plot(x_vals, y_vals_sixteenQAM, label='16QAM', marker='o')

		plt.title("Data Throughput by Modulation with 16QAM and 8PSK Jammed", fontsize=18)
		plt.xlabel('SNR of Eve')
		plt.ylabel('AB Data Throughput')
		plt.ylim(0,4)
		plt.grid(True)
		plt.legend(loc='lower left')

		plt.show()






	def save_figure(self, figure_number, figure_title, file_name):

		plt.figure(figure_number)
		plt.title(figure_title, fontsize=18)
		plt.xlabel('SNR')
		plt.ylabel('Data Throughput')
		plt.ylim(0.0,2.5)
		plt.grid(True)
		plt.savefig(file_name + '_y_axis_0.0-2.5.png', format='png', dpi=300)


		"""
		def save_figure(self, figure_number, figure_title, file_name):
        plt.figure(figure_number)
        plt.xlabel('E_s/N_0 (dB)', fontsize=18)
        plt.ylabel('log_10(BER)', fontsize=18)
        plt.xlim(min(ESNO_RANGE), max(ESNO_RANGE))
        plt.ylim(np.amin(self.ber), 0)
        plt.legend(loc='lower left')
        plt.title(figure_title, fontsize=18)
        plt.grid(True)
        plt.show()
        plt.savefig(file_name + '.eps', format='eps', dpi=1000)
        plt.savefig(file_name + '.png', format='png', dpi=300)
		"""


if __name__ == '__main__':
	main_class = Plot_Datarate_vs_SNR()