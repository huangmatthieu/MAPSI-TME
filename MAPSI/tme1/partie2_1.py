# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 23:49:01 2014

@author: huangmatthieu
"""
import numpy as np
import matplotlib.pyplot as plt

m1 = np.ones((10,1)) * np.array([1,2,3]) # Attention, produit matriciel
m2 = np.ones((10,3)) * 2                 # multiplication par un scalaire
m3 = m1*m2
print m3