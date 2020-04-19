# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:18:01 2020

@author: 766810
"""

test_string=input("Enter string:")
l=[]
l=test_string.split()
wordfreq=[l.count(p) for p in l]
print(dict(zip(l,wordfreq)))
