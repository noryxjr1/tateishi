"""This class for management of input data"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from datetime import date
import pickle as pkl

import util.common_func as cf

class CointFeatureVectorManager(object):
    def __init__(self, **kwargs):
        self._training_start_date = kwargs['TrainingStartDate']
        self._pred_start_date = kwargs['PredictStartDate']
        self._pred_end_date = kwargs.get('PredictEndDate', 
                                         self._pred_start_date)
        self._maxlen = kwargs.get('MaxLen', None)
        
        #------Label Vector------#
        target_label = kwargs.get('TargetLabel', 'Return')
        is_regression = kwargs.get('IsRegression', True)
        input_vector = kwargs.get('InputData', 
                                  pd.read_csv(kwargs["FilePath"]))
        interval = kwargs.get('interval', 5)
        input_vector = self._convert_date_format(input_vector)
        coint_df = self._get_target_cointvec()
        target_features = self._get_target_feature(coint_df.columns, input_vector.columns)
        #Add feature vector
        if 'pvalue' in input_vector.columns:
            target_features.append('pvalue')

        if 'NFCIINDX Index' in input_vector.columns:
            target_features.append('NFCIINDX Index')

        if 'DBQSGSI Index' in input_vector.columns:
            target_features.append('DBQSGSI Index')

        #ticker_list = np.unique([ticker[-3:] for ticker in np.array(input_vector.set_index('ValueDate').columns)]).astype(object)+'JPY Index'
        #fx_df = pd.read_csv('./input/fx_rate.csv')
        #fx_df = self._convert_date_format(fx_df).set_index('ValueDate')
        
        fx_df = cf.get_fx_rate(start_date=input_vector.ValueDate.iloc[0],
                              end_date=input_vector.ValueDate.iloc[-1],
                              ccy_list=coint_df.columns)
        #coint_df = self._convert_date_format(pd.read_csv('./input/coint_vec.csv'))\
        #               .set_index('ValueDate')\
        #               .query("index < @self._pred_start_date").iloc[-1]
        
        assert fx_df.shape[1] == coint_df.shape[1]
        
        coint_index_df = (np.array(fx_df) * np.array(coint_df)).sum(axis=1)
        coint_index_df = pd.DataFrame(coint_index_df[interval:] - coint_index_df[:-interval],
                                     index = fx_df.index[:-interval],
                                     columns = ['Return'])
        #import pdb;pdb.set_trace()
        if is_regression:
            self._label_vector = coint_index_df
        else:
            self._label_vector = pd.DataFrame(coint_index_df.Return.apply(lambda x: 1 if x>0 else 0))
        
        self._feature_vector = input_vector.set_index('ValueDate')[target_features]\
                                           .dropna(axis=0).query("index >= @self._training_start_date \
                                                                & index <= @self._pred_end_date")
                                           #.drop(target_label, axis=1)\
                                           
        
        #for dropna process
        self._label_vector = self._label_vector.loc[self._feature_vector.index]
        #import pdb;pdb.set_trace()
        self._create_ml_data()
            
        exec_pca = kwargs.pop('ExecPCA', False)
        if exec_pca:#Reduce Feature Vector Dimension
            self._exec_pca()


    def _get_target_feature(self, target_ccy, feature_list):
        ccy_list = np.array([ccy.replace('JPY Index', '') for ccy in target_ccy])
        return [feature for feature in feature_list if feature[-3:] in ccy_list]


    def _get_target_cointvec(self, 
                             pvalue_file='./input/pvalue_matrix.csv', 
                             cointvec_file='./input/coint_weight.csv'):
        pvalue_df = self._convert_date_format(pd.read_csv(pvalue_file)).set_index('ValueDate')
        weight_df = self._convert_date_format(pd.read_csv(cointvec_file)).set_index('ValueDate')
        
        value_date = weight_df.query("index < @self._pred_start_date").index[-1]
        target_ccy = pvalue_df.query("index < @self._pred_start_date").iloc[-1].idxmin()
        coint_df = pd.DataFrame(weight_df.query("index == @value_date")).query(" Portfolio == @target_ccy")
        coint_df = coint_df[['Ccy', 'Weight']].reset_index('ValueDate').pivot(index='ValueDate', columns='Ccy')
        coint_df.columns = target_ccy.split(',')

        return coint_df

    def _convert_date_format(self, input_vector, target_col='ValueDate'):
        if '/' in input_vector.ValueDate.iloc[0]:
            input_vector[target_col] = input_vector[target_col].apply(lambda x: date(int(x.split('/')[0]), 
                                                                                    int(x.split('/')[1]), 
                                                                                    int(x.split('/')[2])))
        else:
            input_vector[target_col] = input_vector[target_col].apply(lambda x: date(int(x.split('-')[0]), 
                                                                                    int(x.split('-')[1]), 
                                                                                    int(x.split('-')[2])))
        return input_vector

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
        stdsc = StandardScaler()
        training_data_df = self._feature_vector.query("ValueDate >= @self._training_start_date \
                                                     & ValueDate < @self._pred_start_date")
        self._training_data = pd.DataFrame(stdsc.fit_transform(training_data_df),
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

        self._pred_data = pd.DataFrame(stdsc.transform(pred_data_df),
                                       index=pred_data_df.index,
                                       columns=training_data_df.columns)
        
        self._pred_label = pd.DataFrame(pred_label_df,
                                        index=pred_label_df.index,
                                        columns=pred_label_df.columns)
                
    def _exec_pca(self):
        pca = Isomap(n_components=int(self._training_data.shape[0]/10))
        #pca = KernelPCA(n_components=int(self._training_data.shape[1]),  kernel='rbf', gamma=20.0)
        stdsc = StandardScaler()
        self._training_data = pd.DataFrame(stdsc.fit_transform(pca.fit_transform(self._training_data)),
                                           index=self._training_data.index)
        self._pred_data = pd.DataFrame(stdsc.transform(pca.transform(self._pred_data)),
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
    input_manager = CointFeatureVectorManager(FilePath="./input/feature_vector_port_W_f6.csv", 
                                 TrainingStartDate=traning_start_date,
                                 PredictStartDate=pred_start_date,
                                 PredictEndDate=pred_end_date)
    input_manager.training_data.to_csv('training.csv')
    input_manager.predict_data.to_csv('prdict.csv')
