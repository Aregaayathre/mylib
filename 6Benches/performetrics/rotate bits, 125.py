
.# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:25:00 2020

@author: rajas
"""

#Write a program to rotate bits of a given number.

num=int(input("enter number\n"))
n=int(input("enter shifted positions\n"))
shift=shift1=num # assigning num to shift,shift1 because of do not distrub the original value
# left shift
for i in range(1,9):
    if i<n+1:
        shift=shift<<1
        
print("left shift: ",shift)

# right shift

for j in range(-1,-9,-1):
    if j>-(n+1):
        shift1=shift1>>1
        
print("right shift: ",shift1)