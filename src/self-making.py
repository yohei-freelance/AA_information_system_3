#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:26:26 2019

@author: yohei
"""

import numpy as np
import matplotlib.pyplot as plt

def multi(mean, sigma):
    z = np.array([np.random.randn() for i in range(2)])
    u, d, v = np.linalg.svd(sigma)
    D = np.diag(np.sqrt(d))
    x = np.dot(D, z)
    y = np.dot(np.linalg.inv(u), x)
    y += mean
    return y

mean = [10, 20]
sigma = np.array([[10, 1],[1, 20]])
fig = plt.figure()
x = np.array([multi(mean, sigma) for i in range(200)])
y = np.array([np.random.multivariate_normal(mean, sigma) for i in range(200)])
plt.plot(x[:, 0], x[:, 1], "+", label="self-making")
plt.plot(y[:, 0], y[:, 1], "+", label="numpy module")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('p5.png')