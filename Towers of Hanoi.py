# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:25:31 2020

@author: 766810
"""

def hanoi(disks, source, auxiliary, target):
    if disks == 1:
        print('Move disk 1 from peg {} to peg {}.'.format(source, target))
        return
 
    hanoi(disks - 1, source, target, auxiliary)
    print('Move disk {} from peg {} to peg {}.'.format(disks, source, target))
    hanoi(disks - 1, auxiliary, source, target)
 
 
disks = int(input('Enter number of disks: '))
hanoi(disks, 'I', 'II', 'III')