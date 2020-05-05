# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import kNN_SearchCV


class ML_kNN(ML_Base):
    cv_model = kNN_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._n_neighbors = kwargs.get('n_neighbors', 15)
        self._weights = kwargs.get('weights', 'distance')

        self._params = kwargs.get('param', {'weights': 'distance',
                                            'algorithm': 'auto',
                                            'leaf_size': 30,
                                            'p': 1
                                            })
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
                self._params = self.tune_param(training_data, training_label, self._is_regression)

        if self._is_regression:
            self._model = KNeighborsRegressor(**self._params)
        else:
            self._model = KNeighborsClassifier(**self._params)

        super().learn(training_data, training_label)

