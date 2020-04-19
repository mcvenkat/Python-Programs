# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:13:14 2020

@author: 766810
"""
elements = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Uut Fl Uup Lv Uus Uuo".split()

import PyPDF2
i=elements.index
def m(w,p=("",[])):
 if not w:return p
 x,y,z=w[0],w[:2],w[:3]
 if x!=y and y in elements:
    a=m(w[2:],(p[0]+y,p[1]+[i(y)]))
    if a:return a
 if x in elements:
    b=m(w[1:],(p[0]+x,p[1]+[i(x)]))
    if b:return b
 if z in elements:
    c=m(w[3:],(p[0]+z,p[1]+[i(z)]))
    if c:return c

f=open('/Users/766810/python/largedictionary.pdf','rb')
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(f) 
for l in f:
 x=m(l[:-1])
 if x:print(x[0],x[1])
f.close()