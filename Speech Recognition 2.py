# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:46:00 2020

@author: 766810
"""

import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source:
    print('Say Something:')
    audio=r.listen(source)
    print('Time Over, Thanks a lot:')
    
    

try:
    print('Text:' +r.recognize_google(audio,language='hi-IN'));

except sr.UnknownValueError: 
    print("Google Speech Recognition could not understand audio") 
      
except sr.RequestError as e: 
    print("Could not request results from Google Speech Recognition service; {0}".format(e)) 
