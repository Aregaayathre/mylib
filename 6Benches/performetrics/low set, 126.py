# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:08:08 2020

@author: rajas
"""

#Write a program to get lowest set bit of a number.

binary_num=list()
num=int(input("enter decimal number\n"))
for i in range(0,8):
    shift=num>>i #code to check value of last bit and append 1 or 0 to list make binary number
    if shift&1:
        binary_num.append(1)
    else:
        binary_num.append(0)   
print("lowest set bit of {} is:{} ".format(num,binary_num[0]))
print("heigest set bit of {} is:{} ".format(num,binary_num[-1]))