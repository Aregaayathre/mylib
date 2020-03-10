# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:18:28 2020

@author: rajas
"""

#You are required to write a program to sort the (name, age, height) tuples by ascending order where name is 
#string, age and height are numbers.
#The tuples are input by console. The sort criteria is:
#1: Sort based on name;
#2: Then sort based on age;
#3: Then sort by score.
#The priority is that name > age > score.
#If the following tuples are given as input to the program:
#Tom,19,80  John,20,90  Jony,17,91  Jony,17,93  Json,21,85
#Then, the output of the program should be:
#[('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]
#[('jok','80','20'),('lol','70','10')]

def Sort_Tuple(tup):
    lst = len(tup)
    for i in range(0,lst):
        for j in range(0,lst-i-1):
            if(tup[j][0] > tup[j+1][0]):
                temp=tup[j]
                tup[j]=tup[j+1]
                tup[j+1]=temp
    return tup


list2=input("enter list of tuples like [('tom','19','80'),('john','20','90')]\n")
l=[]
for tup in list2.split('),('):
    tup=tup.replace(')','').replace('(','')
    l.append(tuple(tup.split(',')))
#print(l)

#tup=[('tom','19','80'),('john','20','90'),('jony','17','91'),('jony','17','93'),('json','21','80')]
print(Sort_Tuple(l))
