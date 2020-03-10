# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:29:06 2020

@author: rajas
"""

#Write a program to check whether a number is even or odd using bitwise operator.

n=int(input("enter a number"))
if n&1:
    print(n,"is an odd number")
else:
    print(n,"is an even number")