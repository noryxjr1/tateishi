# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import HGB_SearchCV

class ML_HistGradientBoost(ML_Base):
    cv_model = HGB_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._params = {'learning_rate': 0.01,
                        'max_iter': 50,
                        'max_leaf_nodes': 31,
                        'min_samples_leaf': 2,
                        'l2_regularization': 0.01,
                        'max_bins': 64,
                        'validation_fraction': 0.1,
                        'tol':1e-7
                        }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def importance(self):
        return self._model.feature_importances_


    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
            self._params = self.tune_param(training_data,training_label, self._is_regression)

        if self._is_regression:
            self._model = HistGradientBoostingRegressor(**self._params)
        else:
            self._model = HistGradientBoostingClassifier(**self._params)
        super().learn(training_data, training_label)

