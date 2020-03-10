# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:35:46 2020

@author: rajas
"""

#Write a program to find maximum between three numbers using conditional operator.

n1 = float(input("enter 1st number"))
n2 = float(input("enter 2nd number"))
n3 = float(input("enter 3rd number"))
if (n1>=n2) and (n1>=n3):
    maximum = n1
elif (n2>=n1) and (n2>=n3):
    maximum = n2
else:
    maximum = n3
print("the maximum number is", maximum)




