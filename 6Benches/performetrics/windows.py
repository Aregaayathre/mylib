# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:11:47 2020

@author: rajas
"""

#windows

window=list(range(0,1000))
for x in range(0,1000):
    window[x]=0
s=0
w=0
for s in range(1,1001):
    for w in range(1,1001):
        if w%s==0:
            if window[w-1]==0:
                window[w-1]=1
            else:
                    window[w-1]=0
        else:
                pass
count=0
for i in range(0,1000):
    if window[i]==1:
        count=count+1
        print((i+1),end=" ")
print("\ntotal open windows are:",count)
print("total closed windows:",(1000-count))