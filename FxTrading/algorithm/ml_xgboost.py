# -*- coding: utf-8 -*-
"""
Created on Tue Jul 2 19:30:00 2019
@author: 09757937
"""
import numpy as np
from xgboost.sklearn import XGBRegressor,XGBClassifier

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import XGB_SearchCV

class ML_XGBoost(ML_Base):
    cv_model = XGB_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._params = {'n_estimators':100,
                        'max_depth': 10,
                        'learning_rate': 0.01,
                        'gamma': 0.1,
                        'min_child_weight': 0.001,
                        'max_delta_step': 1,
                        'subsample': 0.5,
                        'reg_alpha': 0
                        }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    
    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
            self._params = self.tune_param(training_data,training_label, self._is_regression)

        if self._is_regression:
            self._model = XGBRegressor(**self._params)

        else:
            self._model = XGBClassifier(**self._params)
        super().learn(training_data, training_label)


    def predict(self, test_data):
        return super().predict(np.array(test_data))


    def predict_one(self, test_data):
        return super().predict(np.array(test_data))[0]

    def predict_proba(self, test_data):
        return self._model.predict_proba(np.array(test_data))

    def predict_one_proba(self, test_data):
        return self._model.predict_proba(np.array(test_data))[0].tolist()

    @property
    def importance(self):
        return self._model.feature_importances_

