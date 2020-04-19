# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:38:05 2020

@author: 766810
"""

#Python program to scrape website  
#and save quotes from website 
import requests 
from bs4 import BeautifulSoup 
import csv 
  
URL = "http://www.values.com/inspirational-quotes"
#URL = "https://www.geeksforgeeks.org/"
#URL = "http://web.mta.info/developers/turnstile.html"
r = requests.get(URL) 
print("response:",r)
  
soup = BeautifulSoup(r.content, 'html5lib') 
  
quotes=[]  # a list to store quotes 
  
table = soup.find('div', attrs = {'id':'container'}) 
  
for row in table.find('div', attrs = {'class':'quote'}): 
    quote = {} 
    quote['theme'] = row.h5.text 
    quote['url'] = row.a['href'] 
    quote['img'] = row.img['src'] 
    quote['lines'] = row.h6.text 
    quote['author'] = row.p.text 
    quotes.append(quote) 
  
filename = 'inspirational_quotes.csv'
with open(filename, 'wb') as f: 
    w = csv.DictWriter(f,['theme','url','img','lines','author']) 
    w.writeheader() 
    for quote in quotes: 
        w.writerow(quote) 
