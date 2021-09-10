#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

### Base data ###
data = np.random.normal(0, 1, 1000)
data = np.sort(data)
data = data[499:999]

#plt.plot(data)
#plt.show()

### Derivative ###
data_d = []
for i in range(0, len(data)-1):
	data_d.append(data[i+1] - data[i])
	
#plt.plot(data_d)
#plt.show()

### First linear fit ###
x1 = np.arange(0, 500)
m1 = np.mean(data_d[0:249])
b1 = data[0]

y1 = m1*x1 + b1

#plt.plot(y1)
#plt.show()

### Second linear fit ###
x2 = np.arange(450, 500)
m2 = np.mean(data_d[490:499])
b2 = data[499] - m2*499

y2 = m2*x2 + b2

#plt.plot(x2, y2)
#plt.show()

### All together ###

plt.plot(data)

plt.plot(y1, color='r')
plt.plot(x2, y2, color='r')

plt.show()

for i in range(0, len(data)):
	print("(" + str(i) + "," + str(data[i]) + ")")

for i in range(0, len(y1)):
	print("(" + str(i) + "," + str(y1[i]) + ")")

for i in range(0, len(y2)):
	print("(" + str(i+450) + "," + str(y2[i]) + ")")

