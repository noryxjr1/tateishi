# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import AB_SearchCV

class ML_Adaboost(ML_Base):
    cv_model = AB_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._depth = kwargs.pop('Depth', 10)
        self._params = kwargs.get('param', {'learning_rate':0.01,
                                            'n_estimators':100})
        if not self._is_regression:
            self._params['algorithm'] = 'SAMME'
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
                self._params = self.tune_param(training_data, training_label, self._is_regression)
                self._params['random_state'] =np.random.RandomState(1) 
        if self._is_regression:
            self._params['base_estimator'] = DecisionTreeRegressor(max_depth=self._depth)
            self._model = AdaBoostRegressor(**self._params)
        else:
            self._params['base_estimator'] = DecisionTreeClassifier(max_depth=self._depth)
            self._model = AdaBoostClassifier(**self._params)
        super().learn(training_data, training_label)

    @property
    def importance(self):
        return self._model.feature_importances_

