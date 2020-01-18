#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:07:02 2019

@author: yohei
"""

import numpy as np
import csv
from scipy.stats import norm
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

T=np.zeros(50)
with open('ballthrow.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        T[i] += float(row[0])
        i += 1

def LL(x):
    a = -sum([np.log(x[0]*1/np.sqrt(2*np.pi*x[3])*np.exp(-(T[i]-x[1])**2/(2*x[3]))+(1-x[0])*1/np.sqrt(2*np.pi*x[4])*np.exp(-(T[i]-x[2])**2/(2*x[4])))for i in range(50)])
    return a

def dif_ev():
    limit = [(0, 1), (0, 50), (0, 50), (0, 200), (0, 200)]
    result = differential_evolution(LL, limit)
    print(result.fun)
    print(result.x)
    return result.x

def gauss_dist(meter, x):
    return [x[0]*norm.pdf(x=meter, loc=x[1], scale=np.sqrt(x[3])) \
            +(1-x[0])*norm.pdf(x=meter, loc=x[2], scale=np.sqrt(x[4])),
            x[0]*norm.pdf(x=meter, loc=x[1], scale=np.sqrt(x[3])),
            (1-x[0])*norm.pdf(x=meter, loc=x[2], scale=np.sqrt(x[4]))]
    
def plot(x):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("record [m]")
    ax.set_ylabel("number [person]")
    ax.hist(T, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    meter_list = np.linspace(0, 60, 100)
    ax.plot(meter_list, 250 * gauss_dist(meter_list, x)[0], label="Gauss Distribution")
    ax.plot(meter_list, 250 * gauss_dist(meter_list, x)[1], label="Female")
    ax.plot(meter_list, 250 * gauss_dist(meter_list, x)[2], label="Male")
    ax.legend()
    plt.savefig("p4.png")

lista = [0]

def step1(x):
    temp = np.array([[(lambda y: x[0] if y == 0 else 1 - x[0])(j)
    * norm.pdf(x=T[i], loc=x[j + 1], scale=np.sqrt(x[j + 3])) /
    (x[0] * norm.pdf(x=T[i], loc=x[1], scale=np.sqrt(x[3])) +
     (1 - x[0]) * norm.pdf(x=T[i], loc=x[2], scale=np.sqrt(x[4])))
    for i in range(len(T))] for j in range(2)]).T
    return temp

def step2(x, temp):
    N_F = sum(temp[:, 0])
    N_M = sum(temp[:, 1])
    muf_next, mum_next = 1 / N_F * sum(temp[:, 0] * T), 1 / N_M * sum(temp[:, 1] * T)
    sigmaf_next,sigmam_next  = 1 / N_F * sum(temp[:, 0] * ((T - muf_next) ** 2)), 1 / N_M * sum(temp[:, 1] * ((T - mum_next) ** 2))
    piF = 1 / len(T) * sum(temp[:, 0])
    x_new = [piF, muf_next, mum_next, sigmaf_next, sigmam_next]
    return x_new

def iterate(x):
    temp = step1(x)
    x = step2(x, temp)
    likeli_val = LL(x)
    lista.append(likeli_val)
    return x, likeli_val

x = [0.5, 15, 30, 20, 165]
for i in range(1000):
    x, likeli_val = iterate(x)
print(x)
print(dif_ev())
plot(x)