# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:21:48 2020

@author: rajas
"""

#Write a program to flip bits of a binary number using bitwise operator.

decimal_num= int(input("enter number"))
binary_num=[]
flip_num=[]
# function to convert decimal to binary
def decimal_binary(decimal_num):
    for i in range(0,8):
        shift=decimal_num>>i  # code to check value of last bit and append 1 or 0 to list make biary number
        if shift&1:
            binary_num.append(1)
        else:
            binary_num.append(0)
    return binary_num.reverse()

# function to flip bits
def flip_binary(decimal_num):
    for i in range(0,8):
        shift=decimal_num>>i  # code to check value of last bit and flip it 
        if shift&1:
            flip_num.append(0)
        else:
            flip_num.append(1)
    return flip_num.reverse()
decimal_binary(decimal_num) # function call to decimal_binary
for j in range(0,8):
       print(binary_num[j],end="")

print("\n after fliping \n")
flip_binary(decimal_num) # fucntion call to flip bits
for j in range(0,8):
       print(flip_num[j],end="")