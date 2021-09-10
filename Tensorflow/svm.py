#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import import_export as iex
import numpy as np
import sys
import pandas as pd

### Parameter ###
if len(sys.argv) < 6:
	print("Not enough arguments!")
	sys.exit()

dimension = sys.argv[5]

### Fetch dataset ###
## Trainset ##
trainset = np.array(iex.import_data(sys.argv[1]))
trainset = trainset.astype('float32') / 2 ** 15

## Trainlabel ##
trainlab = np.array(iex.import_data(sys.argv[2]))

## Queryset ##
queryset = np.array(iex.import_data(sys.argv[3]))
queryset = queryset.astype('float32') / 2 ** 15

## Querylabel ##
querylab = np.array(iex.import_data(sys.argv[4]))

## Set parameter ##
datasize = len(trainset[0])
items = len(trainset)

### Build a model ###
# SVM
svm = tf.keras.Sequential([tf.keras.Input(shape=(datasize,)),
	tf.keras.layers.experimental.RandomFourierFeatures(output_dim=int(dimension), kernel_initializer="laplacian", trainable=True),
	tf.keras.layers.Dense(units=2, activation='sigmoid'),])
		
svm.compile(optimizer="adam", loss="hinge",)

svm.fit(trainset, tf.keras.utils.to_categorical(trainlab), epochs=20, batch_size=256, shuffle=True)

### Save the model ###
converter = tf.lite.TFLiteConverter.from_keras_model(svm)
tflite_model = converter.convert()

# Save the model.
with open('svm.tflite', 'wb') as f:
  f.write(tflite_model)

svm.summary();

### Create confidential interval ###
predict = svm.predict(trainset)

ff_var = []
fp_var = []

for i in range(0, items):
	if trainlab[i] < 0.5:
		ff_var.append(predict[i][0])
		fp_var.append(predict[i][1])

fp_var = np.sort(fp_var)
ff_var = np.sort(ff_var)

### Query the model ###
predict = svm.predict(queryset)

pp_bound = fp_var[int(np.floor(len(fp_var)*0.99))]
ff_bound = ff_var[int(np.ceil(len(ff_var)*0.001))]

print("PP boundary: " + str(pp_bound) + " FF boundary: " + str(ff_bound))

anomaly_count = 0
error_count = 0

pp_query = []
pf_query = []

ff_query = []
fp_query = []

for i in range (0, items):
	if (predict[i][0] < ff_bound and predict[i][1] > pp_bound) or predict[i][0] < predict[i][1]:
		if querylab[i] > 0.5:
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

csv_line.append(str(pp_bound))
csv_line.append(str(ff_bound))

dataframe = pd.DataFrame([csv_line])
dataframe.to_csv("svm_n" + str(dimension) + ".csv", mode="a", index=False, header=False)

