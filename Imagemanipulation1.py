# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:53:12 2020

@author: 766810
"""

from PIL import Image
indeximage=Image.open("Bodies of Water.PNG")
print(indeximage.size)
print(indeximage.format)
newImage=Image.new("RGBA",(500,500),"purple")
newImage.save("newImage.PNG")
indeximage.rotate(90).save("indeximage90.PNG")
indeximage.rotate(180).save("indeximage180.PNG")
indeximage.transpose(Image.FLIP_TOP_BOTTOM).save("indeximagetrans.PNG")

