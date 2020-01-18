#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:11:58 2019

@author: yohei
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

own1 = np.zeros((2,2))
own2 = np.zeros((2,1))

X, Y = [], []
data1, data2 = [], []
with open('Hubbles_constant.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data1.append(float(row[0]))
        data2.append(float(row[1]))
        own1[0][0] += 1
        own1[0][1] += float(row[0])
        own1[1][0] += float(row[0])
        own1[1][1] += float(row[0])**2
        own2[0][0] += float(row[1])
        own2[1][0] += float(row[0])*float(row[1])
        X.append(float(row[0]))
        Y.append(float(row[1]))
    own1_inv = np.linalg.inv(own1)
    answer = np.dot(own1_inv, own2)
    nown = own2[1][0]/own1[1][1]

def own(x):
    temp = answer[0]+answer[1]*x
    return temp

def noown(x):
    return nown*x

def var(datax, datay):
    temp1, temp2 = 0, 0
    for i in range(len(datax)):
        temp1 += (own(datax[i]) - datay[i])**2
        temp2 += (noown(datax[i]) - datay[i])**2
    return temp1/len(datax), temp2/len(datax)

var1, var2 = var(data1, data2)
AIC1 = len(data1)*np.log(var1[0]) + 2*2
AIC2 = len(data1)*np.log(var2) + 2*1
AICc1 = len(data1)*np.log(var1[0]) + 2*2*len(data1)/(len(data1)-2*2-1)
AICc2 = len(data1)*np.log(var2) + 2*1*len(data1)/(len(data1)-2*1-1)

x = np.linspace(0, 2.1, 100)
y1 = own(x)
y2 = noown(x)
plt.plot(x, y1, label="have y-intercept")
plt.plot(x, y2, label="have no y-intercept")
plt.plot(X, Y, '+')
plt.xlim(0, 2.1)
plt.ylim(-300, 1300)
plt.xlabel("Distance [Mpc]")
plt.ylabel("The recession speed [km/s]")
plt.legend()
#plt.show()
plt.savefig('p2.png')