# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 19:30:00 2018
@author: jpbank.quants
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import RF_SearchCV

class ML_RandomForest(ML_Base):
    cv_model = RF_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._params = {'bootstrap':True,
                        #'criterion':'gini',
                        'max_depth' : 15,
                        'max_features':5,
                        'min_samples_leaf':2,
                        'min_samples_split': 3,
                        'n_estimators': 300
                        }
        if not self._is_regression:
            self._params['criterion'] = 'gini'

        if np.isnan(self._params['max_depth']):
            self._params['max_depth'] = None

        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def importance(self):
        return self._model.feature_importances_

    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
                self._params = self.tune_param(training_data, training_label, self._is_regression)
        self._params['random_state'] = 1
        self._params['n_jobs'] = -1
        if self._is_regression:
            #self._model = RandomForestRegressor(criterion='mse',
            #                                    random_state=1,
            #                                    n_jobs=-1,
            #                                    n_estimators=self._params['n_estimators'],
            #                                    max_depth=self._params['max_depth'],
            #                                    max_features=self._params['max_features'])
            self._model = RandomForestRegressor(**self._params)
        else:
            
            
            self._model = RandomForestClassifier(**self._params)
        
        super().learn(training_data, training_label)


    
