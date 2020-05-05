# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
from sklearn.linear_model import LinearRegression, Ridge, Lasso, \
                                 ElasticNet, BayesianRidge, ARDRegression

from algorithm.ml_base import ML_Base

class ML_LinearRegression(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = LinearRegression()
        super().learn(training_data, training_label)
    


class ML_RidgeRegression(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = Ridge()
        super().learn(training_data, training_label)


class ML_LassoRegression(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = Lasso(alpha=0.1)
        super().learn(training_data, training_label)

class ML_ElasticNet(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = ElasticNet(alpha=0.1, l1_ratio=0.7)
        super().learn(training_data, training_label)

class ML_BasianRegression(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = BayesianRidge()
        super().learn(training_data, training_label)
        
class ML_ARDRegression(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def learn(self, training_data, training_label, tunes_param=False):
        self._model = ARDRegression(threshold_lambda=1e5)
        super().learn(training_data, training_label)

