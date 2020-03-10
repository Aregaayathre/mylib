# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:33:17 2020

@author: rajas
"""

#Write a program to convert decimal to binary number system using bitwise operator.

def dectobinary(n):
    for i in range(31,-1,-1):
        k=n>>i
        if (k&1):
            print("1",end="")
        else:
            print("0",end="")
n=5
dectobinary(n)