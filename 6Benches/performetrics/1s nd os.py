# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:20:45 2020

@author: rajas
"""

#Write a program to count total zeros and ones in a binary number.
count_zero=0
count_one=0
binary_num=list()
decimal_num= int(input("enter number"))

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
for j in range(0,8):
       print(binary_num[j],end="")
length=len(binary_num)
for i in range(0,length): # loop to count zeros  and ones
    if binary_num[i]==1:
        count_one=count_one+1
    else:
        count_zero=count_zero+1
print("\n {}  0's and {}  1's in given 8bit binary number ".format(count_zero,count_one))