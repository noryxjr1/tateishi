# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 19:30:00 2018
@author: jpbank.quants
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import numpy as np
import pandas as pd

from sklearn.decomposition import KernelPCA, PCA
from abc import ABCMeta, abstractmethod


class ML_Base(metaclass=ABCMeta):
    #cv_model = None
    def __init__(self, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._is_regression = kwargs.get('IsRegression', True)
        self._with_grid_cv = kwargs.get('with_grid_cv', False)
        self._model = None

        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    #ToDo:Add Decorator
    def learn(self, training_data, training_label):
        if type(training_label) == pd.DataFrame:
            self._model = self._model.fit(np.array(training_data), np.array(training_label).T[0])
        else:
            self._model = self._model.fit(np.array(training_data), np.array(training_label))


    def predict(self, test_data):
        return self._model.predict(test_data)


    def predict_one(self, test_data):
        return self._model.predict(test_data)[0]

    def predict_proba(self, test_data):
        return self._model.predict_proba(test_data)

    def predict_one_proba(self, test_data):
        return self._model.predict_proba(test_data)[0].tolist()

    def dispose(self):
        self._model = None
        del self._model
    
    #@classmethod
    #def tune_param(cls, training_data, training_label, is_regression, with_grid_cv=False):
    #    if cls.cv_model is None:
    #        return {}
    #    else:
    #        rand_search = cls.cv_model(is_regression=is_regression,
    #                                   with_grid_cv=with_grid_cv)
    #        best_param = rand_search.execute(np.array(training_data),
    #                                         np.array(training_label).T[0])
    #        #cls.cv_model.dispose()
    #        return best_param

