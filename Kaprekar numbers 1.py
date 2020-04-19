# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:06:31 2020

@author: 766810
"""

def print_Kaprekar_nums(start, end):
   for i in range(start, end + 1):
      # Get the digits from the square in a list:
      sqr = i ** 2
      digits = str(sqr)


      # Now loop from 1 to length of the number - 1, sum both sides and check
      length = len(digits)
      for x in range(1, length):
         left = int("".join(digits[:x]))
         right = int("".join(digits[x:]))
         if (left + right) == i:
            print("Number: " + str(i) + "Left: " + str(left) + " Right: " + str(right))

#print_Kaprekar_nums(150, 8000)
print_Kaprekar_nums(100, 15000)
