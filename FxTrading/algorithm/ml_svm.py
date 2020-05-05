# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:30:00 2018
@author: jpbank.quants
"""
import sklearn.svm as svm

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import SVM_SearchCV

class ML_SVM(ML_Base):
    cv_model = SVM_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._params = kwargs.get('param', {'kernel':'rbf',
                                            'C':10,
                                            'gamma':1,
                                            'degree': 10})
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    def learn(self, training_data, training_label, tunes_param=False):
        if tunes_param:
            self._params = self.tune_param(training_data, training_label, self._is_regression)

        if self._is_regression:
            self._model = svm.SVR(**self._params)

            #scale training label (predicted result is all zero without this -> F-measure illed) 
            super().learn(training_data, training_label*10)
        else:
            self._params['probability']=True
            self._model = svm.SVC(**self._params)
            super().learn(training_data, training_label)
    
