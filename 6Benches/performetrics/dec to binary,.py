# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:19:30 2020

@author: rajas
"""

#write prgm to convert decimal to binary number sysytem using bitwise operator
binary_num=list()
decimal_num=int(input("enter number"))
for i in range(0,8):
    shift=decimal_num>>i #code to check value of last bit and append 1 or 0 to list make binary number
    if shift&1:
        binary_num.append(1)
    else:
        binary_num.append(0)
for j in range(-1,-9,-1):
    print(binary_num[j],end="")
