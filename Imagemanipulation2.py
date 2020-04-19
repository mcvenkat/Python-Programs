# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:14:19 2020

@author: 766810
"""

from PIL import Image, ImageFilter
#Read image
im = Image.open( 'Bodies of Water.jpg' )
#Display image
im.show()

#Applying a filter to the image
im_sharp = im.filter( ImageFilter.SHARPEN )
#Saving the filtered image to a new file
im_sharp.save( 'image_sharpened.jpg', 'JPEG' )

#Splitting the image into its respective bands, i.e. Red, Green,
#and Blue for RGB
r,g,b = im_sharp.split()

#Viewing EXIF data embedded in image
exif_data = im._getexif()
exif_data