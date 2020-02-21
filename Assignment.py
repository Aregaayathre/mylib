#Write a program to perform input/output of all basic data types.
a = "Hello World"       #string 
b = 15                  #int
c = 35.5                #float
d = 3j                  #complex
e = ["red","Green","purple"]        #list
f = ("black","yellow","orange")     #tuple
g = range(3)                        #range
h = {"car" : "BMW", "series" : "X5"}   #dict
i = {"bmw","audi","jaguar"}             #set
j = True                            #bool
print(a,b,c,d,e,f,g,h,i,j)
print(type(a),type(b),type(c),type(d),type(e),type(f),type(g),type(h),type(i),type(j))

#Write a  program to enter two numbers and find their sum.
num1 = 5
num2 = 5.7
sum = num1 + num2   #add two numbers
print(sum)           #display the sum

#Write a  program to enter two numbers and perform all arithmetic operations.
x = 10
y = 6
print(x + y)     #Addition
print(x - y)        #Subtraction
print(x * y)        #Multiplication
print(x / y)        #division
print(x % y)        #Modulus
print(x ** y)       #Exponentation
print(x // y)       #Floor Division

#Write a  program to enter length and breadth of a rectangle and find its perimeter.
length = float(input('enter the length of the rectangle'))      #enter the length
width = float(input('enter the width of the rectangle'))        #enter the width
perimeter = 2*(length + width)          #calculate the perimeter
print(perimeter)                        #print the perimeter

#Write a  program to enter length and breadth of a rectangle and find its area.
width = float(input('enter the width of the rectangle'))        #enter the width
height = float(input('enther the height of the rectangle'))     #enter the height
area = width * height                       #calculate the are
print(area)

#Write a  program to enter radius of a cirle and find its diameter, cirumference and area.
pi = 3.14                                   #entering pi value
radius = float(input('enter the radius of the circle'))     #enter radius value
diameter = 2 * radius                       #calculating diameter
print(diameter)
circumference = 2 * pi * radius             #calculating circumference
print(circumference)
area = pi * radius * radius                 #calculating are
print(area)

#Write a  program to enter length in centimeter and convert it into meter and kilometer.
cm = float(input("enter value:"))           #enter value in cm
meter = cm / 100                            #convert the cm into meter
km = cm / (1000 * 100)                      #converting cm into km                
print(meter)
print(km)

#Write a  program to enter temperature in celsius and convert it into Fahrenheit.
celsius = 37.5                              #enter the value in celsius
F = (celsius * 1.8) + 32                    #coverting celsius to fahrenheit
print(F)

#Write a  program to enter temperature in Fahrenheit and convert to celsius
f = 35                                      #enter the value in fahrenheit
celsius = (f - 32) / 1.8                    #coverting fahrenheit to celsius
print(celsius)

#Write a  program to convert days into years, weeks and days.
n=int(input("enter noof days in a year"))
year = int(n / 365)  
weeks = int((n % 365) /7)  
days = (n % 365) % 7 
print("years = ",year, "week = ",weeks,"days = ",days)

#Write a  program to find power of any number x ^ y.
x = float(input("enter base number"))
y = float(input("enterpower value"))
z = x ** y
print(z)

#Write a  program to enter any number and calulate its square root.
number = int(input("enter a number: "))
sqrt = number ** 0.5
print("square root:", sqrt)

#Write a  program to enter two angles of a triangle and find the third angle.
a = float(input('enter the first angle of a triangle: '))
b = float(input('enter the second angle of a triangle: '))
c = 180 - (a + b)
print("third angle of a triangle:", c)

#Write a  program to enter base and height of a triangle and find its area.
b = float(input('Enter base of a triangle: '))
h = float(input('Enter height of a triangle: '))
area = (b * h) / 2
print('area of the triangle is:', area)

#Write a  program to calculate area of an equilateral triangle.
import math
side = float(input('Enter length of any side of an Equilateral Triangle: '))
def area_equilateral( side ): 
    area = (sqrt(3) / 4) * side * side 
    print ("Area of Equilateral Triangle:", area) 

#Write a  program to enter marks of five subjets and calculate total, average and perentage.
english = float(input("enter English Marks: "))
maths = float(input("enter Math score: "))
computers = float(input("enter Computer Marks: "))
physics = float(input("enter Physics Marks: "))
chemistry = float(input("enter Chemistry Marks: "))
total = english + maths + computers + physics + chemistry
print("total marks:", total)
average = total / 5
print("average marks:", average)
percentage = (total / 500) * 100
print("percentage:",percentage)

#Write a  program to enter P, T, R and calculate Simple Interest.
p = 5                               #principle amount
t = 1                               #time
r = 4                               #rate
SI = (p * r * t) / 100              #calculating the simple intrest
print("simple intrest:", SI)        #printing the SI value

#Write a  program to enter P, T, R and calculate compound Interest.
p = float(input("enter principle amount"))
t = float(input("enter time period"))
r = float(input("enter rate"))
CI = p*(1+r/100)**t
print("compound Interest:", CI)
