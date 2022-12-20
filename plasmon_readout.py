# -*- coding: utf-8 -*-
"""
@author : Jonathan Hoh
Initial template for photo-mixer readout on ROACH2 system with 512MHz digizers

"""
import casperfpga
import time
import matplotlib.pyplot as plt
import struct
import numpy as np
import csv
import scipy.stats as stats

# Establish parameters and constant based on hardware design
f_s = 500*10**3 # Sampling freq in Hertz
fft_len = 2**9

katcp_port=7147
roach = '192.168.40.79'
#firmware_fpg = 'liss_gold_v1.fpg'
firmware_fpg = 'lock_in_gold_enhanced.fpg'
fpga = casperfpga.katcp_fpga.KatcpFpga(roach, timeout = 3.)
time.sleep(1)
if (fpga.is_connected() == True):
	print ('Connected to the FPGA ')
else:
	print ('Not connected to the FPGA')

stime = time.time()
fpga.upload_to_ram_and_program(firmware_fpg)
print ('Using Jon Hohs Proprietary Black-Magic Optimization Algorithm \nFPGA @ address %s programmed in %.2f seconds \n\n Jons withcraft reduced program time by %.2f seconds' % (fpga.host, time.time() - stime, 31.89*(time.time() - stime)))
time.sleep(1)
print ('\n NAND Gate Flash Success')
time.sleep(1)
print('\nFirware ready for execution')
# Initializing registers

fpga.write_int('fft_shift', 2**9)
fpga.write_int('cordic_freq', 1) # 
fpga.write_int('cum_trigger_accum_len', 2**23-1) # 2**19/2**9 = 1024 accumulations
fpga.write_int('cum_trigger_accum_reset', 0) #
fpga.write_int('cum_trigger_accum_reset', 1) #
fpga.write_int('cum_trigger_accum_reset', 0) #
fpga.write_int('start_dac', 0) #
fpga.write_int('start_dac', 1) #

plt.ion()

def plotFFT():
		fig = plt.figure()
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(0,1024,2), np.zeros(1024/2), '#FF4500', alpha = 0.8)
		line1.set_marker('.')
		plt.grid()
		plt.ylim(-10, 100)
		plt.tight_layout()
		count = 0
		stop = 1.0e6
		while(count < stop):
			fpga.write_int('fft_snap_fft_snap_ctrl',0)
			fpga.write_int('fft_snap_fft_snap_ctrl',1)
			fft_snap = (np.fromstring(fpga.read('fft_snap_fft_snap_bram',(2**9)*8),dtype='>i2')).astype('float')
			I0 = fft_snap[0::4]
			Q0 = fft_snap[1::4]
			mag0 = np.sqrt(I0**2 + Q0**2)
			mag0 = 20*np.log10(mag0)
			line1.set_ydata(mag0)
			fig.canvas.draw()
			count += 1
		return

def plotAccum():
		# Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
		fig = plt.figure(figsize=(10.24,7.68))
		plt.title('TBD, Accum. Frequency = ')# + str(accum_freq), fontsize=18)
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(1016),np.ones(1016), '#FF4500')
		line1.set_linestyle('None')
		line1.set_marker('.')
		plt.xlabel('Channel #',fontsize = 18)
		plt.ylabel('dB',fontsize = 18)
		plt.xticks(np.arange(0,1016,100))
		plt.xlim(0,1016)
		#plt.ylim(-40, 5)
		plt.ylim(-40, 100)
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0
		stop = 10000
		while(count < stop):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mags =(np.sqrt(I**2 + Q**2))[:1016]
			#mags = 20*np.log10(mags/np.max(mags))[:1016]
			mags = 10*np.log10(mags+1e-20)[:1016]
			line1.set_ydata(mags)
			fig.canvas.draw()
			count += 1
		return

def read_accum_snap():
		# 2**9 64bit wide 32bits for mag0 and 32bits for mag1    
		fpga.write_int('accum_snap1_accum_snap_ctrl', 0)
		fpga.write_int('accum_snap1_accum_snap_ctrl', 1)
		accum_data = np.fromstring(fpga.read('accum_snap1_accum_snap_bram', 16*2**9), dtype = '>i').astype('float')
		I = accum_data[0::2]
		Q = accum_data[1::2]
		return I, Q

def bin_reading(bin):
	i = 0
	i_vec = []
	q_vec = []
	while i < 1000:
		I,Q = read_accum_snap()
		i_vec.append(I[bin])
		q_vec.append(Q[bin])
		i += 1
	plt.plot(i_vec)
	return i_vec, q_vec

def plotADC():
		# Plots the ADC timestream
		fig = plt.figure(figsize=(10.24,7.68))
		plot1 = fig.add_subplot(211)
		line1, = plot1.plot(np.arange(0,2048), np.zeros(2048), 'r-', linewidth = 2)
		plot1.set_title('I', size = 20)
		plot1.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plot2 = fig.add_subplot(212)
		line2, = plot2.plot(np.arange(0,2048), np.zeros(2048), 'b-', linewidth = 2)
		plot2.set_title('Q', size = 20)
		plot2.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0
		stop = 1.0e8
		while count < stop:
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 0)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 1)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 0)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_trig', 1)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_trig', 0)
			time.sleep(0.1)
			adc = (np.fromstring(fpga.read('adc_snap_adc_snap_bram',(2**10)*8),dtype='>h')).astype('float')
			adc /= (2**15)
			adc *= 550.
			I = np.hstack(zip(adc[0::4],adc[2::4]))
			Q = np.hstack(zip(adc[1::4],adc[3::4]))
			line1.set_ydata(I)
			line2.set_ydata(Q)
			fig.canvas.draw()
			count += 1
		return

def findmaxbin():
	I, Q = read_accum_snap()
	I = I[2:]
	Q = Q[2:]
	mags =(np.sqrt(I**2 + Q**2))[:1016]
	#mags = 20*np.log10(mags/np.max(mags))[:1016]
	mags = 10*np.log10(mags+1e-20)[:1016]
	max_bin = np.argmax(mags)
	max_val = np.max(mags)
	#print('Maximum power of %d dBW at bin %d'%(max_val, max_bin))
	return max_val, max_bin

def dataCollectSimp(chan, lines):
	# In its current iteration, 10 seconds of data are printed per line
	seconds_per_line = 10
	count1 = 0
	rate = 16
	file = open('spec_data_%d.csv'%(chan), 'w')
	writer = csv.writer(file)
	
	cols = rate * seconds_per_line
	tau = np.logspace(-1, 3, 50)
	writer.writerow([chan])
	
	while (count1 < lines):
		print('we are %d/%d of the way through this shit'%(count1,lines))	    
		vals = np.zeros(cols)
		count2 = 0
		while (count2 < cols):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mag =(np.sqrt(I**2 + Q**2))[chan]
			accum_data = 10*np.log10(mag+1e-20)
			vals[count2] = accum_data
			#print('this is column number %d with a value of %d'%(count2, val))
			count2 += 1
		writer.writerow(vals)
		count1 += 1
	file.close()

def dataCollect4Chan(chan1, chan2, chan3, chan4, lines):
	# In its current iteration, 10 seconds of data are printed per line
	seconds_per_line = 10
	runtime = lines * seconds_per_line
	count1 = 0
	rate = 16 # This is the accumulation frequency, in the full-scale case, 16 Hz
	
	# Open up a file to save CSV
	#  data to, give it unique name based on runtime and channels
	file = open('%d_sec_LISS_accum_%d_%d_%d_%d.csv'%(runtime, chan1,chan2,chan3,chan4), 'w')
	writer = csv.writer(file)
	cols = rate * seconds_per_line
	
	# Create a header row in CSV file with channel names
	writer.writerow([chan1, chan2, chan3, chan4])
	# Iterate through the rows of the CSV file
	while (count1 < lines):
		print('we are %d/%d of the way through this shit'%(count1,lines))	    
		vals1 = np.zeros(cols)
		vals2 = np.zeros(cols)
		vals3 = np.zeros(cols)
		vals4 = np.zeros(cols)
		count2 = 0

		# Iterate through the columns of the CSV file
		while (count2 < cols):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mags = []
			accum_data = [0,0,0,0]
			mags.append((np.sqrt(I**2 + Q**2))[chan1])
			mags.append((np.sqrt(I**2 + Q**2))[chan2])
			mags.append((np.sqrt(I**2 + Q**2))[chan3])
			mags.append((np.sqrt(I**2 + Q**2))[chan4])
			#mags = 20*np.log10(mags/np.max(mags))[:1016]     
			accum_data[0] = 10*np.log10(mags[0]+1e-20)
			accum_data[1] = 10*np.log10(mags[1]+1e-20)
			accum_data[2] = 10*np.log10(mags[2]+1e-20)
			accum_data[3] = 10*np.log10(mags[3]+1e-20)

			# val1-val4 are the accumulation magnitude values for each of the chosen channels
			(val1, val2, val3, val4) = (accum_data[chan1], accum_data[chan2], accum_data[chan3], accum_data[chan4])
			
			# vals1-vals4 are all of the accum magnitudes for the given channel in a single row of data
			vals1[count2] = accum_data[0]
			vals2[count2] = accum_data[1]
			vals3[count2] = accum_data[2]
			vals4[count2] = accum_data[3]
			
			#print('this is column number %d with a value of %d'%(count2, val))
			# count2 will iterate until it reaches the column max, which here is 160 (equivalent to 10 seconds of data)
			count2 += 1 # iterate by 1 column

		# Once all of the columns for a single row are collected, the code dumps those values for all 4 bins to CSV file
		writer.writerow(vals1)
		writer.writerow(vals2)
		writer.writerow(vals3)
		writer.writerow(vals4)
		
		# Iterate through another line of the CSV file and repeat column population
		count1 += 1 # Iterations will continue until reaching the desired run-time of simulation
	
	file.close()
def sync_len_2_time(length):
	accum_time_sec = (length/fft_len)/f_s
	return accum_time_sec

# Define a function that allows Python to quickly morph the accumulator sync trigger system to match desired integration time
def struct_morph(bit_length):
	fpga.write_int('cum_trigger_accum_len', (2**bit_length)-1) 
	fpga.write_int('cum_trigger_accum_reset', 0) #
	fpga.write_int('cum_trigger_accum_reset', 1) #
	fpga.write_int('cum_trigger_accum_reset', 0) #
	return

def is_detectable(pwr_in, b_o_i):
	# This function will act as the main conduit for testing the minimum detectable power of the LISS firmware 
	# It is designed for a single-tone input and will provide the output power in the B.O.I. for a given input power 
	# For each input power level, a number of different integration times will be tested 
	file = open('min_pwr_test_%d_input.csv'%(pwr_in), 'w')
	writer = csv.writer(file)
	powers = np.arange(17,25) # range of accumulator sync lengths in base 2 bits
	taus = sync_len_2_time(2**powers[:]) # Convert accum_len values into mirror time values (in seconds)

	# Say for example you have an input power of -30dB 
	# Calling this function with said power as the argument will cycle the accumulator length through the powers array...
	# and find the power at the B.O.I.
	# It will also check to make sure that the maximum power is in the B.O.I. and if not, will return a false in the output
	
	itr = 0 
	
	output_pwr = np.zeros([len(taus), 3])
	
	# The output_pwr array will contain all of the goodies resulting from this function
	# It is comprised of n columns where n = number of tested integration times
	# as well as 3 rows: the first being the integration time, the second being the output power, and the third being whether it is detectable; [tau, pwr_out, detectable]
	# An example column would be [0.065536, 66.5, 1]; note that the final column element while type int represents a Boolean where 0 = False and 1 = True
	while itr < len(taus):
		struct_morph(powers[itr])

		# Sometimes the firmware needs to activate the accumulator a couple of times before it gets the correct readings
		# Make a routine to run find_max_bin 5 times and take the mode of the output values
	
		max_bins = np.arange(0,5)
		max_vals = np.arange(0,5)

		for i in max_bins:
			max_vals[i], max_bins[i] = findmaxbin()
		
		# Take the mode bin and mean value as a broad litmus test on efficacy
		mode_bin = stats.mode(max_bins)
		mean_val = np.median(max_vals)

		if mode_bin == b_o_i:
			detectable = True
			print('Max power detected at bin %d with average power of %d dBW'%(mode_bin, mean_val))
			print('With an input power of %d and integration time of %d ms, signal is detected'%(pwr_in, taus[itr]))
		else:
			detectable = False
			print('Max power detected at bin %d instead of estimated bin of %d'%(mode_bin, b_o_i))
			print('With an input power of %d and integration time of %d ms, signal is NOT detected'%(pwr_in, taus[itr]))
		
		output_pwr[itr] = [taus[itr], mean_val, detectable]

		# At this point, we have successfully tested a tau element at a given input power and have the resulting output
		# Only thing left to do is iterate to the next integration time and test again
		# And I guess write everything to a CSV file for the sake of completion

		writer.writerow(output_pwr[itr])
		itr += 1
		print('Iteration %d of %d for input power of %d'%(itr, len(taus)-1, pwr_in))
	
	print('for input power %d, output powers versus integration times are:'%(pwr_in))
	print(output_pwr)
	return output_pwr




	
	
