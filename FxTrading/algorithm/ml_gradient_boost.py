# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import GB_SearchCV

class ML_GradientBoost(ML_Base):
    cv_model = GB_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._n_estimators = kwargs.pop('n_estimators', 300)
        self._max_depth = kwargs.pop('max_depth', 15)
        if np.isnan(self._max_depth):
            self._max_depth = None

        self._params = {'criterion':'mse',
                        'learning_rate':0.01,
                        #'loss' : 'deviance',
                        'max_features':5,
                        'min_samples_leaf': 2,
                        'min_samples_split': 3,
                        'min_weight_fraction_leaf': 0.001
                        }
        if not self._is_regression:
            self._params['loss'] = 'deviance'
        
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def importance(self):
        return self._model.feature_importances_


    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
                self._params = self.tune_param(training_data, training_label, self._is_regression)
        if self._is_regression:
            self._model = GradientBoostingRegressor(**self._params)
        else:
            self._model = GradientBoostingClassifier(**self._params)
        super().learn(training_data, training_label)

