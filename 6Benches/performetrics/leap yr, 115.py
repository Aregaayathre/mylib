# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:47:39 2020

@author: rajas
"""

#Write a program to check whether year is leap year or not using conditional operator

year = int(input("enter a year"))
if (year%4)==0:
    if (year%100)==0:
        if (year%400)==0:
            print("{0}is a leap year".format(year))
        else:
            print("{0}is not a leap year".format(year))
    else:
            print("{0}is a leap year".format(year))
else:
            print("{0}is not a leap year".format(year))