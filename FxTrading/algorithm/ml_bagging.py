# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:34:00 2018
@author: jpbank.quants
"""
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import BG_SearchCV

class ML_Bagging(ML_Base):
    cv_model = BG_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._params = {'bootstrap':True,
                        'max_features':5,
                        'max_samples' : 10,
                        'n_estimators':300
                        }
        self._depth = kwargs.pop('Depth', 10)

        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
            self._params = self.tune_param(training_data, training_label, self._is_regression)
        if self._is_regression:
            self._params['base_estimator'] = DecisionTreeRegressor(max_depth=self._depth)
            self._model = BaggingRegressor(**self._params)
        else:
            self._params['base_estimator'] = DecisionTreeClassifier(max_depth=self._depth)
            self._model = BaggingClassifier(**self._params)
        super().learn(training_data, training_label)

    @property
    def importance(self):
        return self._model.feature_importances_

