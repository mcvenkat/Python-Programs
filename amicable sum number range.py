# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:05:15 2020

@author: 766810
"""

def amicable_sum(num):
	divisor_sum = [0] * num
	for i in range(1, len(divisor_sum)):
		for j in range(i * 2, len(divisor_sum), i):
			divisor_sum[j] += i	
        # Find all amicable pairs 
	result = 0
	for i in range(1, len(divisor_sum)):
		j = divisor_sum[i]
		if j != i and j < len(divisor_sum) and divisor_sum[j] == i:
			result += i
	return str(result)

print(amicable_sum(5000))
print(amicable_sum(100000))