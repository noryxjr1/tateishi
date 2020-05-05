"""This class is to create new feature"""
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import date
import copy

import util.common_func as cf


class FeatureVectorMaker(object):
    def __init__(self, **kwargs):
        self._base_vector = kwargs.pop('BaseVector',
                                       pd.read_csv('./input/Qlearning_weekly.csv'))
        self._base_vector['ValueDate'] = pd.to_datetime(self._base_vector.ValueDate.values)#cf.convert_date_format(self._base_vector.ValueDate)
        self._base_vector = self._base_vector[['ValueDate', 'Return']].set_index('ValueDate')

        self._feature_vector = kwargs.pop('FeatureVector', 
                                          pd.read_csv('./input/Qlearning_daily.csv'))
        self._feature_vector['ValueDate'] = pd.to_datetime(self._feature_vector.ValueDate.values)#cf.convert_date_format(self._feature_vector.ValueDate)
        self._feature_vector = self._feature_vector.drop(['Return'],axis=1).set_index('ValueDate')

        self._new_feature = copy.deepcopy(self._feature_vector)

        change_param_list = [5, 21, 63]
        std_param_list = [5, 21, 63]
        skew_param_list = [5, 21, 63]
        kurt_param_list = [5, 21, 63]
        mv_avg_param_list = [5, 21, 63]

        print("Calculating Moving Average Distance...")
        self.add_feature(self.add_moving_avg_dist, mv_avg_param_list)

        print("Calculating Change Rate...")
        self.add_feature(self.add_change, change_param_list)

        print("Calculating Standard Deviation...")
        self.add_feature(self.add_std, std_param_list)

        print("Calculating Skew...")
        self.add_feature(self.add_skew, skew_param_list)

        print("Calculating Kurtosis...")
        self.add_feature(self.add_kurt, kurt_param_list)

        print("Calculating Moving Average...")
        self.add_feature(self.add_moving_avg, mv_avg_param_list)

    @property
    def feature_vector(self):
        self._new_feature = self._new_feature.drop(self._feature_vector.columns, 
                                                   axis=1).fillna(0.0)
        return pd.merge(self._new_feature, self._base_vector, right_index=True, left_index=True)

    def add_feature(self, target_func, param_list):
        for p in param_list:
            self._new_feature = pd.merge(self._new_feature, 
                                         target_func(p),
                                         right_index=True,
                                         left_index=True)

    def add_change(self, param=5):
        
        change_matrix = pd.DataFrame([(self._feature_vector.iloc[i] \
                                     - self._feature_vector.iloc[i-param])\
                                    / (np.abs(self._feature_vector.iloc[i]) \
                                    + np.abs(self._feature_vector.iloc[i-param]))
                                      for i in range(param, self._feature_vector.shape[0])])

        change_matrix.index = self._feature_vector.index[param:]
        change_matrix.columns = [i + '_Change{0}'.format(param) \
                                for i in self._feature_vector.columns]

        return change_matrix
            

    def add_std(self, param=5):
        std_df = pd.DataFrame([np.std(self._feature_vector.iloc[i-param:i],ddof=1) \
                               for i in range(param, self._feature_vector.shape[0]+1)])
        std_df.index = self._feature_vector.index[param-1:]
        std_df.columns = [i + '_Std{0}'.format(param) \
                          for i in self._feature_vector.columns]

        return std_df


    def add_skew(self, param=5):
        skew_df = pd.DataFrame([self._feature_vector.iloc[i-param:i].skew() \
                               for i in range(param, self._feature_vector.shape[0]+1)])
        skew_df.index = self._feature_vector.index[param-1:]
        skew_df.columns = [i + '_Skew{0}'.format(param) \
                          for i in self._feature_vector.columns]

        return skew_df


    def add_kurt(self, param=5):
        kurt_df = pd.DataFrame([self._feature_vector.iloc[i-param:i].kurt() \
                               for i in range(param, self._feature_vector.shape[0]+1)])
        kurt_df.index = self._feature_vector.index[param-1:]
        kurt_df.columns = [i + '_Kurt{0}'.format(param) \
                          for i in self._feature_vector.columns]

        return kurt_df
    

    def add_moving_avg(self, param=5):
        mv_avg_df = self._feature_vector.rolling(param).mean().dropna(axis=0)
        mv_avg_df.columns = [i + '_MvAvg{0}'.format(param) \
                            for i in self._feature_vector.columns]

        return mv_avg_df


    def add_moving_avg_dist(self, param=5):
        mv_avg__dit_df = self._feature_vector.iloc[param-1:] \
                       - self._feature_vector.rolling(param).mean().dropna(axis=0)
        mv_avg__dit_df.columns = [i + '_MvAvgDist{0}'.format(param) \
                            for i in self._feature_vector.columns]

        return mv_avg__dit_df

if __name__ == '__main__':
    feature_maker = FeatureVectorMaker()

    import os
    from Config.MLConfigParser import MLConfigParser
    config = MLConfigParser()
    feature_maker.feature_vector.to_csv(os.path.join(config.input_dir, 
                                                      config.feature_file))
    #with open(os.path.join(config.input_dir, 
    #                       config.feature_file),'wb') as f:
    #    pkl.dump(feature_maker.feature_vector, f, protocol=-1)



