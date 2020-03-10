# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:18:24 2020

@author: rajas
"""


#write an prgm which accepts a sequence of comma seperated 4 digit binary numbers as its input and then check whether they r divisible by 5 or not

items=[]
num=[x for x in input().split(',')]
for p in num:
    x=int(p,2)
    if x%5==0:
        items.append(p)
print(",".join(items))





