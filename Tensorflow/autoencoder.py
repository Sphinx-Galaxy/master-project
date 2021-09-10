#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import import_export as iex
import numpy as np
import os
import sys
import statistics
import pandas as pd
import matplotlib.pyplot as plt

### Parameter ###
if len(sys.argv) < 5:
	print("Not enough arguments!")
	sys.exit()

encoding_dim = int(sys.argv[4])

### Fetch datasets ###
## Trainset ##
trainset = np.array(iex.import_data(sys.argv[1]))
trainset = trainset.astype('float32') / 2 ** 15

## Queryset ##
queryset = np.array(iex.import_data(sys.argv[2]))
queryset = queryset.astype('float32') / 2 ** 15

## Querylabel ##
querylab = np.array(iex.import_data(sys.argv[3]))

## Set parameter ##
datasize = len(trainset[0])
items = len(trainset)

### Build autoencoder ###
input_vec = tf.keras.Input(shape=(datasize,))

encoded = tf.keras.layers.Dense(encoding_dim, activation='tanh')(input_vec)
decoded = tf.keras.layers.Dense(datasize, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(input_vec, decoded)

## Encoder ##
encoder = tf.keras.Model(input_vec, encoded)

## Decoder ##
encoded_input = tf.keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.Model(encoded_input, decoder_layer(encoded_input))

### Train the model ###
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(trainset, trainset, epochs=5, batch_size=256, shuffle=True)
	
### Save the model ###
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the model.
with open('aec.tflite', 'wb') as f:
  f.write(tflite_model)

autoencoder.summary();

### Create confidential interval ###
predict = autoencoder.predict(trainset)

## Create difference between training and prediction ##
diff = []
for i in range(0, items):
	diff.append(np.sum(np.absolute(np.absolute(predict[i]) - np.absolute(trainset[i]))))
	
### Sort difference ##
diff = np.sort(diff)

### Derive the diff ##
derv = []

for i in range(0, len(diff)-1):
	derv.append(diff[i+1] - diff[i])

### Calculate lineare regression ###
m1 = np.mean(derv[0:int(len(diff)*0.5)])
b1 = diff[0]

m2 = np.mean(derv[int(len(diff)*0.99):])
b2 = diff[-1] - m2 * len(diff)

y1 = m1 * np.arange(0, len(diff)) + b1
y2 = m2 * np.arange(0, len(diff)) + b2

### Get the boundary
boundary = diff[int((b1-b2)/(m2-m1))]
print("Boundary: " + str(boundary))

### Query the model ###
predict = autoencoder.predict(queryset)
anomaly_count = 0
error_count = 0

for i in range (0, items):
	sum_pq = np.sum(np.absolute(np.absolute(predict[i]) - np.absolute(queryset[i])))

	if sum_pq > boundary:
		if querylab[i] == 1:
			anomaly_count += 1
			
		else:
			error_count += 1

print("Result: ")
print("Total anomalies: " + str(anomaly_count) + " out of " + str(np.sum(querylab)))
print("Positive-Positive-Rate: " + str(100 * anomaly_count / np.sum(querylab)))
print("Total errors: " + str(error_count) + " out of " + str(len(querylab)))
print("False-Positive-Rate: " + str(100 * error_count / len(querylab)))

#### Write results to csv ###
csv_line = []
index = ((sys.argv[1].split('_')[-1]).split('.bin')[0])[1:]
csv_line.append(index) # trainset
csv_line.append(str(anomaly_count / np.sum(querylab)))
csv_line.append(str(error_count / len(querylab)))

dataframe = pd.DataFrame([csv_line])
dataframe.to_csv("autoencoder_n" + str(encoding_dim) + ".csv", mode="a", index=False, header=False)
