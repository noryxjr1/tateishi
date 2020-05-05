# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: nory.xjr1
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

from algorithm.ml_base import ML_Base

class ML_NaiveBayes(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, training_data, training_label,  tunes_param=False):
        self._model = GaussianNB()
        if np.any(training_label<0):
            training_label = training_label.Return.apply(lambda x: 1 if x>0 else 0)
        super().learn(training_data, training_label)
        #self._model.fit(training_data, training_label)

