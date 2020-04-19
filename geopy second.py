# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:13:04 2020

@author: 766810
"""

#from geopy.distance import vincenty
from geopy import distance

ig_gruendau = (50.195883, 9.115557)
delphi = (49.99908,19.84481)

#print(vincenty(ig_gruendau,delphi).miles)
#print(geopy.distance.geodesic(ig_gruendau,delphi).miles)
wellington = (-41.32, 174.81)
salamanca = (40.96, -5.50)
print(distance.distance(wellington, salamanca).km)
