# -*- coding: utf-8 -*-
"""
Performetrics Assignments
"""

"""
Write a program to find maximum between three numbers using conditional operator
"""
a, b, c = 10, 20, 15
if a > b and a > c:
    print ("maximum number a:", a)
elif b > a and b > c:
    print("maximum number b:", b)
else:
    print("maximum number c:", c)
    
'''
Write a program to check whether year is leap year or not using conditional operator
'''
year = int(input("Enter a year: "))
if (year % 4) == 0:
    if (year % 100 == 0):
        if (year % 400 == 0):
            print("%d is a leaf year" %year)
        else:
            print("%d is not an leaf year" %year)
    else:
        print("%d is a leaf year" %year)
else:
    print("%d is not an leaf year" %year)
    
'''
Write a program to check whether character is an alphabet or not using conditional operator
'''
ch = input("Enter an character")
if((ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z')):
    print(ch, "is an Alphabet")
else:
    print(ch, "is not an Alphabet")
    
'''
Write a program to find maximum between two numbers with/without using conditional operator
'''
a = int(input("Enter a value: "))
b = int(input("Enter b value: "))
if b > a:
    print("b is greater than a")
else:
    print("a is greater than b")
    
'''
Write a program to swap two numbers using bitwise operator
'''
a = int(input("enter a value: "))
b = int(input("enter b value: "))
print("before swapping:  a=", a, " b=", b)
a = a ^ b
b = a ^ b
a = a ^ b
print("\nafter swapping: a=", a, " b=", b)

'''
Write a program to check whether a number is even or odd using conditional operator
'''
num = int(input("enter a num : "))
if(num%2 == 0):
    print("given num is even")
else:
    print("given num is odd")

'''
Write a program to check whether a number is even or odd using bitwise operator
'''
num = int(input("enter a num : "))
if num & 1 :
    print(num, "is an odd num")
else:
    print(num, "is an even num")
