# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:12:14 2020

@author: 766810
"""

from sklearn.datasets import make_regression, make_classification
X,y=make_regression()
from sklearn import dummy
fakeestimator = dummy.DummyRegressor(strategy='median')
fakeestimator.fit(X,y)
print (fakeestimator.predict(X)[:5])
