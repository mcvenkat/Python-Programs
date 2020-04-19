# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:11:29 2020

@author: 766810
"""
# Python3 program to count  
# amicable pairs in an array 

# Calculate the sum  
# of proper divisors 
def sumOfDiv(x): 
    sum = 1
    for i in range(2, x): 
        if x % i == 0: 
            sum += i 
    return sum
  
# Check if pair is amicable 
def isAmicable(a, b): 
    if sumOfDiv(a) == b and sumOfDiv(b) == a: 
        return True
    else: 
        return False
  
# This function prints pair  
# of amicable pairs present  
# in the input array 
def countPairs(arr, n): 
    count = 0
    for i in range(0, n): 
        for j in range(i + 1, n): 
            if isAmicable(arr[i], arr[j]): 
                count = count + 1
    return count 

# Driver Code 
arr1 = [220, 284, 1184, 
        1210, 2, 5] 
n1 = len(arr1) 
print(countPairs(arr1, n1)) 
  
arr2 = [2620, 2924, 5020,  
        5564, 6232, 6368] 
n2 = len(arr2) 
print(countPairs(arr2, n2))     
