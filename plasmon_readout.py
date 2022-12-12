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


katcp_port=7147
roach = '192.168.40.79'
firmware_fpg = 'liss_no_mag_2022_Dec_05_0404.fpg'
#firmware_fpg = 'lock_in_v1_2021_Mar_25_1417.fpg'
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

def dataCollect4Chan(chan1, chan2, chan3, chan4, lines):
	runtime = lines * 10
	count1 = 0
	rate = 16
	#file = open('%d_sec_lock_in_accum_%d_%d_%d_%d.csv'%(runtime, chan1,chan2,chan3,chan4), 'w')
	file = open('%d_sec_hoh_spec_accum_%d_%d_%d_%d.csv'%(runtime, chan1,chan2,chan3,chan4), 'w')
	writer = csv.writer(file)
	seconds_per_line = 10
	cols = rate * seconds_per_line
	tau = np.logspace(-1, 3, 50)
	writer.writerow([chan1, chan2, chan3, chan4])
	while (count1 < lines):
		print('we are %d/%d of the way through this shit'%(count1,lines))	    
		vals1 = np.zeros(cols)
		vals2 = np.zeros(cols)
		vals3 = np.zeros(cols)
		vals4 = np.zeros(cols)
		count2 = 0
		while (count2 < cols):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mags =(np.sqrt(I**2 + Q**2))[:1016]
			#mags = 20*np.log10(mags/np.max(mags))[:1016]
			mags = 10*np.log10(mags+1e-20)[:1016]
			accum_data = 10*np.log10(mags+1e-20)[:1016]
			#print(mags)
			(val1, val2, val3, val4) = (accum_data[chan1], accum_data[chan2], accum_data[chan3], accum_data[chan4])
			vals1[count2] = val1
			vals2[count2] = val2
			vals3[count2] = val3
			vals4[count2] = val4
			#print('this is column number %d with a value of %d'%(count2, val))
			count2 += 1
		writer.writerow(vals1)
		writer.writerow(vals2)
		writer.writerow(vals3)
		writer.writerow(vals4)
		count1 += 1
	file.close()