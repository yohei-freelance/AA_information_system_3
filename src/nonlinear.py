#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 06:22:29 2019

@author: yohei
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

t = np.arange(0,14000,280)
T = np.zeros(50)
with open('satellite-temp.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        T[i] += float(row[1])
        i += 1

x0 = np.array([[np.mean(T)],[2.0],[2.0*np.pi/6000],[0],[0.1]])
x1 = np.array([[0],[0],[0],[0],[0]])
n = T.size

while (np.linalg.norm(x1-x0) > 0.00001):
    A = np.zeros([5,5])
    f_x = np.zeros([5,1])
    A[0][0] = 2 * n
    for i in range(n):
        A[0][1] += 2*np.sin(x0[2]*t[i]+x0[3])
        A[0][2] += 2*x0[1]*t[i]*np.cos(x0[2]*t[i]+x0[3])
        A[0][3] += 2*x0[1]*np.cos(x0[2]*t[i]+x0[3])
        A[1][1] += 2*np.sin(x0[2]*t[i]+x0[3])**2
        A[1][2] += 2*t[i]*(x0[0]-T[i])*np.cos(x0[2]*t[i]+x0[3])+2*x0[1]*t[i]*np.sin(2*(x0[2]*t[i]+x0[3]))
        A[1][3] += 2*(x0[0]-T[i])*np.cos(x0[2]*t[i]+x0[3])+2*x0[1]*np.sin(2*(x0[2]*t[i]+x0[3]))
        A[2][2] += 2*x0[1]*t[i]**2*(T[i]-x0[0])*np.sin(x0[2]*t[i]+x0[3])+2*x0[1]**2*t[i]**2*np.cos(2*(x0[2]*t[i]+x0[3]))
        A[2][3] += 2*x0[1]*t[i]*(T[i]-x0[0])*np.sin(x0[2]*t[i]+x0[3])+x0[1]**2*t[i]*np.cos(2*(x0[2]*t[i]+x0[3]))
        A[3][3] += 2*x0[1]*(T[i]-x0[0])*np.sin(x0[2]*t[i]+x0[3])+2*x0[1]**2*np.cos(2*(x0[2]*t[i]+x0[3]))
        A[4][0] += 1.0/2/x0[4]**2*(-2)*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        A[4][1] += 1.0/2/x0[4]**2*(-2)*np.sin(x0[2]*t[i]+x0[3])*(T[i]- x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        A[4][2] += 1.0/2/x0[4]**2*(-2)*x0[1]*t[i]*np.cos(x0[2]*t[i]+x0[3])*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        A[4][3] += 1.0/2/x0[4]**2*(-2)*x0[1]*np.cos(x0[2]*t[i]+ x0[3])*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        A[4][4] += 1.0/2/x0[4]**2-1.0/x0[4]**3*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))**2
        f_x[0] += -2*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        f_x[1] += -2*np.sin(x0[2]*t[i]+x0[3])*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        f_x[2] += -2*x0[1]*t[i]*np.cos(x0[2]*t[i]+x0[3])*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        f_x[3] += -2*x0[1]*np.cos(x0[2]*t[i]+x0[3])*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))
        f_x[4] += -1.0/2/x0[4]+1.0/2/x0[4]**2*(T[i]-x0[0]-x0[1]*np.sin(x0[2]*t[i]+x0[3]))**2
    A[1][0] = A[0][1]
    A[2][0] = A[0][2]
    A[3][0] = A[0][3]
    A[2][1] = A[1][2]
    A[3][1] = A[1][3]
    A[3][2] = A[2][3]
    A = np.linalg.inv(A)
    x1 = x0
    x0 = x1 - np.dot(A,f_x)

T1 = x0[0] + x0[1]*np.sin(x0[2]*t+x0[3])
print (x0)

plt.plot(t,T,"+")
plt.plot(t,T1, label="optimized model")
plt.xlabel("Time [sec]")
plt.ylabel("Temperature [degree]")
plt.legend()
#plt.show()
plt.savefig('p3.png')