# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:00:13 2020

@author: 766810
"""

from PyPDF2 import PdfFileReader
import slate3k as slate

 
 
def text_extractor(path):
    with open(path, 'rb') as f:
        extracted_text = slate.PDF(f)
        pdf = PdfFileReader(f)
 
        # get the first page
       # page = pdf.getPage(0)
       # print('Page type: {}'.format(str(type(page))))
       # text = page.extractText()
       # print(text)
       # print(extracted_text)
        number_of_pages =pdf.getNumPages()
        print('number of pages:' + str(number_of_pages))
 
        for page in range(pdf.getNumPages()):
            pdf_page = pdf.getPage(page)
            print('Page type: {}'.format(str(type(pdf_page))))
          #  pdf_text = pdf_page.extractText()
            print(extracted_text)
            print('page:' + str(page))
        print('number of pages:' + str(number_of_pages))
     

        
if __name__ == '__main__':
    path = 'Policy 1.pdf'
    text_extractor(path)