# -*- coding: utf-8 -*-

"""
Created on Wed Dec 5 19:30:00 2018
@author: jpbank.quants
"""
import logging
import numpy as np
import pandas as pd
from statsmodels import api as sm
from sklearn.metrics import r2_score, roc_auc_score, \
                            accuracy_score, recall_score, \
                            precision_score, f1_score, \
                            roc_auc_score

class ResultManager(object):
    def __init__(self, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        predicted_df = kwargs['PredictedData']
        label_df = kwargs['PredictedLabel']
        self._multi_class = kwargs.get('multi_class', False)
        self._predict_dic = {}
        for algo in np.unique(predicted_df.Algorithm):
            result_df = pd.merge(pd.DataFrame(predicted_df.query("Algorithm == @algo").Predict),
                                 label_df,
                                 right_index = True,
                                 left_index = True)
            #import pdb;pdb.set_trace()
            result_df.columns = ['PredictedReturn', 'ActualReturn']
            result_df = self._add_class_column(result_df, 'Predicted')
            self._predict_dic[str(algo)] = result_df
            

        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def _add_class_column(self, target_df, column_prefix):
        if self._multi_class:
            target_df[column_prefix+'Class'] = target_df[column_prefix+'Return']
            target_df['ActualClass'] = target_df['ActualReturn']
        else:
            target_df[column_prefix+'Class'] = target_df[column_prefix+'Return']\
                                          .apply(lambda x: 1 if x>0 else 0)
            target_df['ActualClass'] = target_df['ActualReturn']\
                                      .apply(lambda x: 1 if x>0 else 0)

        return target_df


    @property
    def predicted_data(self):
        return self._predict_dic


    def create_result(self, column_prefix='Predicted', type='Predict'):
        pred_result_df = pd.DataFrame()
        algo_list = self._predict_dic.keys()
        for algo in algo_list:
            pred_result_df = pred_result_df.append(
                                self._create_summary_result(
                                self._predict_dic[algo], 
                                    column_prefix,
                                    type))

        pred_result_df['Algorithm'] = algo_list
        return pred_result_df
        

    def _create_summary_result(self, target_df, column_prefix, type):
        pred_reg_array = np.array(np.array(target_df[column_prefix+'Return']))
        actual_reg_array = np.array(target_df['ActualReturn'])
        pred_class_array = np.array(np.array(target_df[column_prefix+'Class']))
        actual_class_array = np.array(target_df['ActualClass'])
        #import pdb;pdb.set_trace()

        if self._multi_class:
            return pd.DataFrame({'LinR2_regression': sm.OLS(pred_reg_array,
                                                            sm.add_constant(actual_reg_array)).fit().rsquared,
                                 'LinR2_class': sm.OLS(pred_class_array,
                                                       sm.add_constant(actual_class_array)).fit().rsquared,
                                 'NonLinR2_regression': r2_score(pred_reg_array, actual_reg_array),
                                 'NonLinR2_class': r2_score(pred_class_array, actual_class_array),
                                 'Accuracy': accuracy_score(actual_class_array, pred_class_array),
                                 'IC_regression':np.corrcoef(pred_reg_array, actual_reg_array)[0,1],
                                 'IC_class':np.corrcoef(pred_class_array, actual_class_array)[0,1],
                                 'Type':type},
                                 index=[0])
        else:
            return pd.DataFrame({'LinR2_regression': sm.OLS(pred_reg_array,
                                                        sm.add_constant(actual_reg_array)).fit().rsquared,
                             'LinR2_class': sm.OLS(pred_class_array,
                                                   sm.add_constant(actual_class_array)).fit().rsquared,
                             'NonLinR2_regression': r2_score(pred_reg_array, actual_reg_array),
                             'NonLinR2_class': r2_score(pred_class_array, actual_class_array),
                             'Accuracy': accuracy_score(actual_class_array, pred_class_array),
                             'IC_regression':np.corrcoef(pred_reg_array, actual_reg_array)[0,1],
                             'IC_class':np.corrcoef(pred_class_array, actual_class_array)[0,1],
                             'Recall': recall_score(actual_class_array, pred_class_array),
                             'Precision': precision_score(actual_class_array, pred_class_array),
                             'F1': f1_score(actual_class_array, pred_class_array),
                             'AUC': roc_auc_score(actual_class_array, pred_class_array),
                             'Type':type},
                             index=[0])



    def create_return_series(self, column_prefix='Predicted', type='Predict'):
        algo_list = self._predict_dic.keys()
        return_series_df = pd.DataFrame()
        for alg in algo_list:
            target_df = self._predict_dic[alg]
            pred_reg_array = np.array(np.array(target_df[column_prefix+'Return']))
            actual_reg_array = np.array(target_df['ActualReturn'])
            return_series_df = return_series_df.append([[np.abs(actual_reg_array[i]) 
                                                        if actual_reg_array[i]*pred_reg_array[i]>0
                                                        else np.abs(actual_reg_array[i])*-1
                                                        for i in range(target_df.shape[0])]])


        #import pdb;pdb.set_trace()
        return_series_df = return_series_df.T
        return_series_df.index = target_df.index
        return_series_df.columns = algo_list

        return return_series_df
