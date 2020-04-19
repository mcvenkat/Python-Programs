# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:24:15 2020

@author: 766810
"""

import re
mobilenumberex=re.compile(r"\d\d\d\d\d\d\d\d\d\d")
mo=mobilenumberex.search("my mobile number is 9940498016")
print(mo.group())

tryRegex=re.compile(r".ap")
print(tryRegex.findall("rap tap map sap cap zap nap"))