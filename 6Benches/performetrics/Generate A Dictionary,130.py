# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:43:17 2020

@author: rajas
"""

#Define a function which can generate a dictionary where the keys are numbers between 1 and 20 (both included)
# and the values are square of keys.The function should print all key/value combinations.
d=dict()
for x in range(1,21):
    d[x]=x**2
print(d)