# -*- coding: utf-8 -*-
"""
Created on Tue Jul 2 19:30:00 2019
@author: 09757937
"""
import numpy as np
import lightgbm as lgb

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import LGBM_SearchCV

class ML_LightGBM(ML_Base):
    cv_model = LGBM_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._params = kwargs.get('param', {'num_leaves': 100,
                                            'max_depth': -1,
                                            'learning_rate':0.01,
                                            'n_estimators':100,
                                            'subsample_for_bin':5000,
                                            'min_split_gain':0.,
                                            'min_child_weight':1e-3,
                                            'min_child_samples':5,
                                            'subsample_freq':0,
                                            'reg_alpha':0.})

        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    
    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
            self._params = self.tune_param(training_data, training_label, self._is_regression)

        if self._is_regression:
            self._model = lgb.LGBMRegressor(**self._params)
        else:
            self._model = lgb.LGBMClassifier(**self._params)
        super().learn(training_data, training_label)


    def predict(self, test_data):
        return super().predict(np.array(test_data))


    def predict_one(self, test_data):
        return super().predict(np.array(test_data))[0]


    @property
    def importance(self):
        return self._model.feature_importances_
    
