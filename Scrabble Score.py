# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:37:22 2020

@author: 766810
"""

# Creates a dictionary called 'score' which takes each letter in scrabble as keys and assigns their respective points as values
score = {"a": 1 , "b": 3 , "c": 3 , "d": 2 ,
         "e": 1 , "f": 4 , "g": 2 , "h": 4 ,
         "i": 1 , "j": 8 , "k": 5 , "l": 1 ,
         "m": 3 , "n": 1 , "o": 1 , "p": 3 ,
         "q": 10, "r": 1 , "s": 1 , "t": 1 ,
         "u": 1 , "v": 4 , "w": 4 , "x": 8 ,
         "y": 4 , "z": 10}

# Creates a function which takes 1 argument
def scrabble_score(word):

# Initial score is 0 
    total = 0;

# Looks at every letter in 'word'
    for i in word:

# Makes all the letters lowercase (since the dictionary 'score' has no uppercase letters due to laziness)
        i = i.lower(); 
        
# Adds the score of each letter up and putting them into 'total'
        total = total + score[i];
    return total;

# Allows you to input a word
your_word = input("Enter a word: ");

# The score of your word is printed
print(scrabble_score(your_word));