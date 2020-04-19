# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:05:18 2020

@author: 766810
"""

# get_doc_info.py
 
from PyPDF2 import PdfFileReader
 
 
def get_info(path):
    with open(path, 'rb') as f:
        pdf = PdfFileReader(f)
        info = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()
 
    print(info)
 
  #  author = info.author
  #  creator = info.creator
  #  producer = info.producer
  #  subject = info.subject
  #  title = info.title
 
if __name__ == '__main__':
    path = 'largedictionary.pdf'
    get_info(path)