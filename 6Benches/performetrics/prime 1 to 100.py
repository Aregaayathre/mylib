# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:35:36 2020

@author: rajas
"""

#prime numbers
for i in range(2,101):
    count=0
    for j in range(2,i+1):
        if(i%j==0):
            count=count+1
    if(count<2):
        print(i,end=" ")