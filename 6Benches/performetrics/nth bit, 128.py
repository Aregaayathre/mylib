# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:03:07 2020

@author: rajas
"""

#Write a program to clear nth bit of a number.

import math
sum=0
binary_num=list()
decimal_num= int(input("enter number"))
n=int(input("enter the position of cler bit\n"))
# function to convert decimal to binary
def decimal_binary(decimal_num):
    for i in range(0,8):
        shift=decimal_num>>i  # code to check value of last bit and append 1 or 0 to list make biary number
        if shift&1:
            binary_num.append(1)
        else:
            binary_num.append(0)
    return binary_num.reverse()

decimal_binary(decimal_num) # function call

length=len(binary_num)
for i in range(-1,-(length+1),-1):
    if i==-(n+1):
        
        if binary_num[i]==1:
            binary_num[i]=0
    
l=0          
for k in range(-1,-(length+1),-1):
    sum+=(binary_num[k]*math.pow(2,l))
    l=l+1

      
print(int(sum))
