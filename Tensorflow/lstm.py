#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import import_export as iex
import numpy as np
import matplotlib.pyplot as plt
import sys 
import pandas as pd

### Parameter ###
if len(sys.argv) < 5:
	print("Not enough arguments!")
	sys.exit()

dimension = float(sys.argv[4])

### Fetch dataset ###
## Trainset ##
trainset = np.array(iex.import_data(sys.argv[1]))
trainset = trainset.astype('float32') / 2 ** 15

train_l = trainset[0:-1] #(trainset.transpose()[0:49]).transpose()
train_r = trainset[1:] #(trainset.transpose()[50:99]).transpose()

## Queryset ##
queryset = np.array(iex.import_data(sys.argv[2]))
queryset = queryset.astype('float32') / 2 ** 15

query_l = queryset[0:-1] #(queryset.transpose()[0:49]).transpose()
query_r = queryset[1:] #(queryset.transpose()[50:99]).transpose()

## Querylabel ##
querylab = np.array(iex.import_data(sys.argv[3]))[0:-1]

## Set parameter ##
datasize = len(trainset[0])
items = len(trainset)-1

### Build a model ###
# LSTM
lstm = tf.keras.Sequential([
#	tf.keras.Input(shape=(len(train_l[0]),)),
	tf.keras.layers.Embedding(10, 10, input_length=len(train_l[0])), #These numbers (10, 10) dont seem to matter
	tf.keras.layers.LSTM(int(dimension)),
	tf.keras.layers.Dense(len(train_r[0]), activation="tanh"),])

lstm.compile(optimizer="adam", loss="mean_squared_error")

lstm.fit(train_l, train_r, epochs=10, batch_size=256, shuffle=True)

### Save the model ###
converter = tf.lite.TFLiteConverter.from_keras_model(lstm)
tflite_model = converter.convert()

# Save the model.
with open('lstm.tflite', 'wb') as f:
  f.write(tflite_model)

lstm.summary();

reference = np.zeros(len(train_l[0]))
### Create a reference curve ###
for i in range(0, items):
	reference += train_l[i]

reference /= items

### Calculate cutoff
predict = lstm.predict(train_l)

## Create difference between training and prediction ##
diff = []
for i in range(0, items):
	diff.append(np.sum(np.absolute(np.absolute(predict[i]) - np.absolute(query_r[i]))))
	
### Sort difference ##
diff = np.sort(diff)
diff = diff[int(len(diff)/2):]

plt.plot(diff)
plt.show()

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
predict = lstm.predict(query_l)

anomaly_count = 0
error_count = 0

for i in range (0, items):
	sum_pq = np.sum(np.absolute(np.absolute(predict[i]) - np.absolute(reference)))	

	if querylab[i] > 0.5:
		for k in range(0, len(predict[i])):
			print("(" + str(k) + ", " +  str(predict[i][k]) + ")")
		
		print()
		
		for k in range(0, len(queryset[i])):
			print("(" + str(k) + ", " +  str(query_l[i][k]) + ")")
			
		plt.plot(query_l[i])
		plt.plot(predict[i])
		plt.show()
		sys.exit()

	if sum_pq > boundary: 
		if querylab[i] > 0.5:
			anomaly_count += 1
		else:
			error_count += 1
			
	
				
print("Result: ")
print("Total anomalies: " + str(anomaly_count) + " out of " + str(np.sum(querylab)))
print("Positive-Positive-Rate: " + str(100 * anomaly_count / np.sum(querylab)))
print("Total errors: " + str(error_count) + " out of " + str(len(querylab)))
print("False-Positive-Rate: " + str(100 * error_count / len(querylab)))

anomaly = 0
nomaly = 0

for i in range (0, items):
	sum_pq = np.sum(np.absolute(np.absolute(predict[i]) - np.absolute(query_r[i])))	
	
	if nomaly < 10 and querylab[i] < 0.5:
		print("Normal: " + str(sum_pq))
		nomaly += 1
			
	if anomaly < 10 and querylab[i] > 0.5:
		print("Anomaly: " + str(sum_pq))
		anomaly += 1

#### Write results to csv ###
csv_line = []
index = ((sys.argv[1].split('_')[-1]).split('.bin')[0])[1:]
csv_line.append(index) # trainset
csv_line.append(str(anomaly_count / np.sum(querylab)))
csv_line.append(str(error_count / len(querylab)))

dataframe = pd.DataFrame([csv_line])
dataframe.to_csv("lstm_b" + str(dimension) + ".csv", mode="a", index=False, header=False)

#num_words = 1000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)

#X_train = sequence.pad_sequences(X_train, maxlen = 200)
#X_test = sequence.pad_sequences(X_test, maxlen = 200)

#model = Sequential()
#model.add(Embedding(num_words, 50, input_length = 200))
#model.add(Dropout(0.2))
#model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
#model.add(Dense(250, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation='sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X_train, y_train, batch_size=64, epochs=10)

#print('\nAccuracy: {}'. format(model.evaluate(X_test, y_test)[1]))
