#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import import_export as iex
import pandas as pd

### Parameter ###
train_probability = 0.01
query_probability = 0.1

low = 0.5
high = 1.0
noise = 0.05 # 0.01 is barley noise, 0.1 is all over the place
dlength = 100
items = 5256
conv = 2 ** 14

build_folder = "./build"
	
### Functions ###

def generate_clean_data(low, high, length, noise):
	#Generate clean sine
	amp = (high - low) / 2
	data = (-1) * np.cos(2 * np.pi * np.arange(0, 1, 1/length)) * amp + amp + low
	
	#Add noise
	data += np.random.normal(0, noise, length)

	return data
	
def add_anomalous_slope(data, length, slope):
	anomaly_point =  random.randint(0, len(data)-length)

	for i in range(anomaly_point, anomaly_point + length):
		data[i] += (i - anomaly_point) * slope
		
	for i in range(anomaly_point + length, len(data)):
		data[i] += length * slope

	return data

def generate_slopeset(low, high, dlength, noise, items, conv, slope):
	slope_train_set = []
	slope_train_lab = []
	slope_query_set = []
	slope_query_lab = []

	## Generate training and query ##
	for i in range(0, items):
		# Training #
		if random.randint(0, int(1/train_probability)) == 0:
			slope_train_set.append(add_anomalous_slope(
				generate_clean_data(low, high, dlength, noise), 50, slope))
			slope_train_lab.append(1)
		else:
			slope_train_set.append(generate_clean_data(low, high, dlength, noise))
			slope_train_lab.append(0)

		# Query #
		if random.randint(0, int(1/query_probability)) == 0:
			slope_query_set.append(add_anomalous_slope(
				generate_clean_data(low, high, dlength, noise), 50, slope))
			slope_query_lab.append(1)
		else:
			slope_query_set.append(generate_clean_data(low, high, dlength, noise))
			slope_query_lab.append(0)
				
	## Export the data ##
	iex.export_data(conv * np.array(slope_train_set), 
		build_folder + "/slope_train_set_m"  + "{0:.4f}".format(slope) + ".bin")
	iex.export_data(slope_train_lab, 
		build_folder + "/slope_train_lab_m"  + "{0:.4f}".format(slope) + ".bin")
	
	iex.export_data(conv * np.array(slope_query_set), 
		build_folder + "/slope_query_set_m"  + "{0:.4f}".format(slope) + ".bin")
	iex.export_data(slope_query_lab, 
		build_folder + "/slope_query_lab_m"  + "{0:.4f}".format(slope) + ".bin")
	
	## Generate example csv ##
	for i in range(0, items):
		if slope_train_lab[i] == 1:
			dataset_to_csv(slope_train_set[i], 
				"slope $m$", "slope_train_set_m"  + "{0:.4f}".format(slope))
			break
	
	for i in range(0, items):
		if slope_query_lab[i] == 1:
			dataset_to_csv(slope_query_set[i], 
				"slope $m$", "slope_query_set_m"  + "{0:.4f}".format(slope))
			break
		
def add_anomalous_pulse(data, length, height):
	anomaly_point =  random.randint(0, len(data)-length)

	for i in range(anomaly_point, anomaly_point + length):
		data[i] *= height

	return data

def generate_pulseset(low, high, dlength, noise, items, conv, height):
	pulse_train_set = []
	pulse_train_lab = []
	pulse_query_set = []
	pulse_query_lab = []

	## Generate training and query ##
	for i in range(0, items):
		# Training #
		if random.randint(0, int(1/train_probability)) == 0:
			pulse_train_set.append(add_anomalous_pulse(
				generate_clean_data(low, high, dlength, noise), 50, height))
			pulse_train_lab.append(1)
		else:
			pulse_train_set.append(generate_clean_data(low, high, dlength, noise))
			pulse_train_lab.append(0)

		# Query #
		if random.randint(0, int(1/query_probability)) == 0:
			pulse_query_set.append(add_anomalous_pulse(
				generate_clean_data(low, high, dlength, noise), 50, height))
			pulse_query_lab.append(1)
		else:
			pulse_query_set.append(generate_clean_data(low, high, dlength, noise))
			pulse_query_lab.append(0)
				
	## Export the data ##
	iex.export_data(conv * np.array(pulse_train_set), 
		build_folder + "/pulse_train_set_h"  + "{0:.3f}".format(height) + ".bin")
	iex.export_data(pulse_train_lab, 
		build_folder + "/pulse_train_lab_h"  + "{0:.3f}".format(height) + ".bin")
	
	iex.export_data(conv * np.array(pulse_query_set), 
		build_folder + "/pulse_query_set_h"  + "{0:.3f}".format(height) + ".bin")
	iex.export_data(pulse_query_lab, 
		build_folder + "/pulse_query_lab_h"  + "{0:.3f}".format(height) + ".bin")
	
	## Generate example csv ##
	for i in range(0, items):
		if pulse_train_lab[i] == 1:
			dataset_to_csv(pulse_train_set[i], 
				"pulse $h$", "pulse_train_set_h"  + "{0:.3f}".format(height))
			break
	
	for i in range(0, items):
		if pulse_query_lab[i] == 1:
			dataset_to_csv(pulse_query_set[i], 
				"pulse $h$", "pulse_query_set_h"  + "{0:.3f}".format(height))
			break
			
def add_anomalous_sine(data, length, period, amp):
	anomaly_point =  random.randint(0, len(data)-length)

	for i in range(anomaly_point, anomaly_point + length):
		data[i] += np.sin(2 * np.pi * (i - anomaly_point) / period) * amp

	return data
			
def generate_sineset(low, high, dlength, noise, items, conv, period, amp):
	sine_train_set = []
	sine_train_lab = []
	sine_query_set = []
	sine_query_lab = []

	## Generate training and query ##
	for i in range(0, items):
		# Training #
		if random.randint(0, int(1/train_probability)) == 0:
			sine_train_set.append(add_anomalous_sine(
				generate_clean_data(low, high, dlength, noise), 50, period, amp))
			sine_train_lab.append(1)
		else:
			sine_train_set.append(generate_clean_data(low, high, dlength, noise))
			sine_train_lab.append(0)

		# Query #
		if random.randint(0, int(1/query_probability)) == 0:
			sine_query_set.append(add_anomalous_sine(
				generate_clean_data(low, high, dlength, noise), 50, period, amp))
			sine_query_lab.append(1)
		else:
			sine_query_set.append(generate_clean_data(low, high, dlength, noise))
			sine_query_lab.append(0)
				
	## Export the data ##
	iex.export_data(conv * np.array(sine_train_set), 
		build_folder + "/sine_train_set_a"  + "{0:.3f}".format(amp) + ".bin")
	iex.export_data(sine_train_lab, 
		build_folder + "/sine_train_lab_a"  + "{0:.3f}".format(amp) + ".bin")
	
	iex.export_data(conv * np.array(sine_query_set), 
		build_folder + "/sine_query_set_a"  + "{0:.3f}".format(amp) + ".bin")
	iex.export_data(sine_query_lab, 
		build_folder + "/sine_query_lab_a"  + "{0:.3f}".format(amp) + ".bin")
	
	## Generate example csv ##
	for i in range(0, items):
		if sine_train_lab[i] == 1:
			dataset_to_csv(sine_train_set[i], 
				"sine $a$", "sine_train_set_a"  + "{0:.3f}".format(amp))
			break
	
	for i in range(0, items):
		if sine_query_lab[i] == 1:
			dataset_to_csv(sine_query_set[i], 
				"sine $a$", "sine_query_set_a"  + "{0:.3f}".format(amp))
			break
			
def dataset_to_csv(dataset, index, filename):
	csv = pd.DataFrame(np.transpose(dataset), columns=["sensor"])
	csv.index.name = index
	
	csv.to_csv(build_folder + "/" + filename + ".csv", mode="w")

if __name__ == "__main__":	
	# Anomalies
	sloperange = np.arange(0.0001, 0.0051, 0.0001) 
	pulserange = np.arange(1.004, 1.204, 0.004)
	sinerange = np.arange(0.003, 0.153, 0.003)
	sineperiod = 10 #[5, 10, 25, 50]

#	generate_slopeset(low, high, dlength, noise, items, conv, 0.01)
#	generate_pulseset(low, high, dlength, noise, items, conv, 1.5)
#	generate_sineset(low, high, dlength, noise, items, conv, 10, 0.1)
	
	for slope in sloperange:
		generate_slopeset(low, high, dlength, noise, items, conv, slope)
		
	for height in pulserange:
		generate_pulseset(low, high, dlength, noise, items, conv, height)
		
	for amp in sinerange:
		generate_sineset(low, high, dlength, noise, items, conv, sineperiod, amp)
