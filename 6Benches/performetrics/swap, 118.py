# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:49:59 2020

@author: rajas
"""

#Write a program to swap two numbers using bitwise operator.

n1=int(input("enter the 1st number:"))
n2=int(input("enter the 1st number:"))
n1=n1^n2
n2=n1^n2
n1=n1^n2
print("swapped value of x is %d & y is %d"%(n1,n2))