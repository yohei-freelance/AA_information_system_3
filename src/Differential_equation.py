#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:23:12 2019

@author: yohei
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 1], [-1, -0.3]])
B = np.array([[0], [1]])
eigenvalue, eigenvector = np.linalg.eig(A)
a, b = np.real(eigenvalue[0]), np.imag(eigenvalue[0])
P = np.array([[1, 1],[eigenvalue[0], eigenvalue[1]]])
P_inv = np.linalg.inv(P)
temp = np.dot(P_inv, A)
D = np.dot(temp, P)

def ideal(t):
    func = 83/78*np.exp(a*t)*(np.cos(b*t)+1249/(1560*b)*np.sin(b*t)) - 5/78*(np.cos(2*t)+5*np.sin(2*t))
    return func

def A_p(delta):
    coef = -np.exp(a*delta)/b
    A_prime = np.array([[coef*(a*np.sin(b*delta)-b*np.cos(b*delta)), coef*(-np.sin(b*delta))],
                        [coef*(np.sin(b*delta)), coef*(-a*np.sin(b*delta)-b*np.cos(b*delta))]])
    return A_prime

def B_p(delta):
    A_inv = np.linalg.inv(A)
    middle_factor = A_p(delta) - np.array([[1,0],[0,1]])
    temp = np.dot(A_inv, middle_factor)
    B_prime = np.dot(temp, B)
    return B_prime

def vectolizer(delta):
    first_vec = np.array([[1],
                        [0]])
    vec_possess = []
    for i in range(int(100/delta)):
        u = np.sin(2*i*1.0/delta)
        next_vec = np.dot(A_p(delta), first_vec) + np.dot(B_p(delta), u)
        vec_possess.append(next_vec)
        first_vec = next_vec
    yn = []
    for i in range(int(100/delta)):
        yn.append(vec_possess[i][0])
    return yn

t = np.linspace(0, 100, 100)
y1 = ideal(t)
y2 = vectolizer(1.0)
plt.plot(t, y1, label="analytical answer")
plt.plot(t, y2, label="delta1.00 answer")
plt.xlabel("time [s]")
plt.ylabel("displacement [m]")
plt.legend()
plt.show()