# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:10:01 2020

@author: rajas
"""

#Write a program to find maximum between two numbers with/without using conditional operator.

n1=float(input("enter 1st number"))
n2=float(input("enter 2nd number"))
max=n1 if n1>n2 else n2
print("maximum number",max)