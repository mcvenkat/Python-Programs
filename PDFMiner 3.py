# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:42:27 2020

@author: 766810
"""

fp = open('/Users/766810/python/Policy 1.pdf','rb')
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine


parser = PDFParser(fp)
doc = PDFDocument()
parser.set_document(doc)
doc.set_parser(parser)
doc.initialize('')
rsrcmgr = PDFResourceManager()
laparams = LAParams()
laparams.char_margin = 1.0
laparams.word_margin = 1.0
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)
extracted_text = ''



for page in doc.get_pages():
    interpreter.process_page(page)
    layout = device.get_result()
    for lt_obj in layout:
        if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            extracted_text += lt_obj.get_text()
            


with open('convertedFile.txt',"wb") as txt_file:
    txt_file.write(extracted_text.encode("utf-8"))
    

import re
with open('convertedFile.txt', 'r') as inF:
    for line in inF:
        # Capture one-or-more characters of non-whitespace after the initial match
        match = re.search(r'Policyno:', line)

        # Did we find a match?
        if match:
            # Yes, process it
            print('Policy no: {}',line)    


