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

'''
Write a program to count total zeros and ones in a binary number
'''
count_zero = 0
count_one = 0
binary_num = list()
decimal_num = int(input("enter number"))


'''
Create a tuple (1,2,3,4,5,6), then remove element 5 from it.
'''
tuple1 = tuple((1,2,3,4,5,6))
print(tuple1)
del tuple1
tuple1 = (1,2,3,4,6)
print(tuple1)

'''
Write a program to toggle nth bit of a number.
'''
num = int(input("number"))
n = int(input("position of bit to toggle"))
print("result after toggling")
print(num ^ (1 << (n-1)))   # xor with the position bit to toggle

'''
You are required to write a program to sort the (name, age, height) tuples by ascending order where name is string, age and height are numbers.
'''
n = int(input("enter the limit"))
list1 = list()
for i in range(0,n):
    print("enter details for {} person", i+1 )
    name = input("enter name")
    age = int(input("enter age"))
    score = int(input("enter score"))
    t1 = (name,age,score)
    list1.append(t1)
print("given list is")
print(list1)

list1.sort()
print("after sorting list is")
print(list1)



















































