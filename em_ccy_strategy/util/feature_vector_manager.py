"""This class for management of input data"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.manifold import Isomap
from datetime import date
import pickle as pkl

import util.common_func as cf

class FeatureVectorManager(object):
    def __init__(self, **kwargs):
        self._training_start_date = kwargs['TrainingStartDate']
        self._pred_start_date = kwargs['PredictStartDate']
        self._pred_end_date = kwargs.get('PredictEndDate', 
                                         self._pred_start_date)
        self._maxlen = kwargs.get('MaxLen', None)
        self._scaler_type = kwargs.get('scaler_type', 1)
        multi_class = kwargs.get('multi_class', False)

        #------Label Vector------#
        target_label = kwargs.get('TargetLabel', 'Return')
        is_regression = kwargs.get('IsRegression', True)
        input_vector = kwargs.get('InputData', 
                                  pd.read_csv(kwargs["FilePath"]))
        
        input_vector['ValueDate'] = pd.to_datetime(input_vector.ValueDate.values)
        
        if is_regression:
            self._label_vector = input_vector\
                                .set_index(['ValueDate'])[[target_label]]
        else:
            if multi_class:
                self._label_vector = pd.DataFrame(input_vector.set_index(['ValueDate'])[target_label])
            else:
                self._label_vector = pd.DataFrame(input_vector
                                                 .set_index(['ValueDate'])[target_label]
                                                 .apply(lambda x: 1 if x>0 else 0))

        self._feature_vector = input_vector.set_index('ValueDate')\
                                           .drop(target_label, axis=1)\
                                           .dropna(axis=0)

        #for dropna process
        self._label_vector = self._label_vector.loc[self._feature_vector.index]
        
        self._create_ml_data()
            
        if kwargs.get('ExecPCA', False):#Reduce Feature Vector Dimension
            self._exec_pca()

        if kwargs.get('select_feature', False):
            self._select_feature()

    @property
    def training_data(self):
        return self._training_data

    @property
    def predict_data(self):
        if self._maxlen is None:
            return self._pred_data
        else:
            return np.array([np.array(self._pred_data)])

    @property
    def training_label(self):
        return self._training_label

    @property
    def predict_label(self):
        if self._maxlen is None:
            return self._pred_label
        else:
            return np.array([np.array(self._pred_label)])


    def _create_ml_data(self):
        if self._scaler_type == 1:
            scaler = StandardScaler()
        elif self._scaler_type == 2:
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler(quantile_range=(25., 75.))

        training_data_df = self._feature_vector.query("ValueDate >= @self._training_start_date \
                                                     & ValueDate < @self._pred_start_date")
        self._training_data = pd.DataFrame(scaler.fit_transform(training_data_df),
                                           index=training_data_df.index,
                                           columns=training_data_df.columns)
        training_label_df = self._label_vector.query("ValueDate >= @self._training_start_date \
                                                    & ValueDate < @self._pred_start_date")
        self._training_label = pd.DataFrame(training_label_df,
                                            index=training_label_df.index,
                                            columns=training_label_df.columns)
        if self._maxlen is None:
            pred_data_df = self._feature_vector.query("ValueDate >= @self._pred_start_date \
                                                     & ValueDate <= @self._pred_end_date")
            pred_label_df = self._label_vector.query("ValueDate >= @self._pred_start_date \
                                                    & ValueDate <= @self._pred_end_date")
        else:
            pred_data_df = self._feature_vector.query("ValueDate <= @self._pred_end_date").iloc[-self._maxlen:]
            pred_label_df = self._label_vector.query("ValueDate <= @self._pred_end_date").iloc[-self._maxlen:]

        self._pred_data = pd.DataFrame(scaler.transform(pred_data_df),
                                       index=pred_data_df.index,
                                       columns=training_data_df.columns)

        self._pred_label = pd.DataFrame(pred_label_df,
                                        index=pred_label_df.index,
                                        columns=pred_label_df.columns)
                
    def _exec_pca(self):
        pca = Isomap(n_components=int(self._training_data.shape[0]/10))
        #pca = KernelPCA(n_components=int(self._training_data.shape[1]),  kernel='rbf', gamma=20.0)
        if self._scaler_type == 1:
            scaler = StandardScaler()
        elif self._scaler_type == 2:
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler(quantile_range=(25., 75.))

        self._training_data = pd.DataFrame(scaler.fit_transform(pca.fit_transform(self._training_data)),
                                           index=self._training_data.index)
        self._pred_data = pd.DataFrame(scaler.transform(pca.transform(self._pred_data)),
                                       index=self._pred_data.index)

    def _select_feature(self):
        if self._scaler_type == 1:
            scaler = StandardScaler()
        elif self._scaler_type == 2:
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler(quantile_range=(25., 75.))

        selector = SelectKBest(k=10, score_func=mutual_info_classif)
        selector.fit(self._training_data, self._training_label)
        self._training_data = pd.DataFrame(scaler.fit_transform(selector.transform(self._training_data)),
                                           index=self._training_data.index)
        self._pred_data = pd.DataFrame(scaler.transform(selector.transform(self._pred_data)),
                                       index=self._pred_data.index)


    def dispose(self):
        self._feature_vector = None
        self._label_vector = None
        self._training_data = None
        self._training_label = None
        self._pred_data = None
        self._pred_label = None
        
        del self._feature_vector
        del self._label_vector
        del self._training_data
        del self._training_label
        del self._pred_data
        del self._pred_label


if __name__ == "__main__":
    traning_start_date = date(2005,1,4)
    pred_start_date = date(2009,1,4)
    pred_end_date = date(2009, 12, 30)
    input_manager = FeatureVectorManager(FilePath="./input/feature_vector.csv", 
                                 TrainingStartDate=traning_start_date,
                                 PredictStartDate=pred_start_date,
                                 PredictEndDate=pred_end_date)
    input_manager.training_data.to_csv('training.csv')
    input_manager.predict_data.to_csv('prdict.csv')
