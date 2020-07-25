# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:40:00 2019

@author: 09757937
"""
import sys
import pandas as pd
import numpy as np
import copy

import util.common_func as cf

class PerformanceMeasurer(object):
    def __init__(self, **kwargs):
        self._monthly_rebalance = kwargs.get('monthly_rebalance', False)
        if not self._monthly_rebalance:
            self._interval = kwargs.get('interval', 1)


    def create_result_summary(self, return_df):
        if self._monthly_rebalance:
            mul = 261
        else:
            mul = 52

        summary_df = pd.DataFrame(return_df.mean(axis=0) * mul).T
        summary_df = summary_df.append(pd.DataFrame(return_df.std(ddof=1, axis=0) * np.sqrt(mul)).T)
        summary_df = summary_df.append(pd.DataFrame(summary_df.iloc[0] / summary_df.iloc[1]).T)
        summary_df = summary_df.append(self._calc_maxdd(return_df))
        summary_df = summary_df.append(pd.DataFrame(summary_df.iloc[0] / summary_df.iloc[3]).T)
        summary_df = summary_df.append(self._calc_var(return_df))
        summary_df = summary_df.append(self._calc_skew_kurt(return_df, calc_skew=True))
        summary_df = summary_df.append(self._calc_skew_kurt(return_df, calc_skew=False))
        summary_df = summary_df.append(self._calc_hit_ratio(return_df))
        summary_df = summary_df.append(self._calc_aggregated_hitratio(return_df, False))
        summary_df = summary_df.append(self._calc_aggregated_hitratio(return_df, True))

        summary_df.index = ['AverageReturn', 
                            'Volatility', 
                            'SharpeRatio', 
                            'MaxDD', 
                            'CalmarRatio',
                            'HVaR',
                            'Skew',
                            'Kurtosis',
                            'DailyHitRatio',
                            'WeeklyHitRatio',
                            'MonthlyHitRatio']
        return summary_df


    def _calc_maxdd(self, return_df):
        cum_return_df = return_df.cumsum(axis=0)
        maxdd_df = pd.DataFrame()
        for col in cum_return_df.columns:
            factor_return_df = cum_return_df[col]
            max_dd = -sys.float_info.max
            for i in range(1, factor_return_df.shape[0] - 1):
                
                if factor_return_df.iloc[i - 1] < factor_return_df.iloc[i] \
                    and factor_return_df.iloc[i] > factor_return_df.iloc[i + 1]:
                    down_rate = factor_return_df.iloc[i] - factor_return_df.iloc[i + 1:].min()
                    
                    if down_rate > max_dd: max_dd = down_rate

            maxdd_df = maxdd_df.append([[col, max_dd]])

        return maxdd_df.set_index(0).T


    def _calc_var(self, return_df, var_param=0.99):
        target_index = max(int(return_df.shape[0] * (1 - var_param)) - 1, 0)
        return pd.DataFrame([np.abs([return_df[[col]].sort_values(by=col)\
                                                     .iloc[target_index][col] \
                                    for col in return_df.columns])],
                            columns = return_df.columns,
                            index = ['HVaR'])

    def _calc_skew_kurt(self, return_df, calc_skew=True):
        if calc_skew:
            return pd.DataFrame([[return_df[col].skew() \
                                  for col in return_df.columns]],
                                columns = return_df.columns,
                                index = ['Skew'])
        else:
            return pd.DataFrame([[return_df[col].kurt() \
                                  for col in return_df.columns]],
                                columns = return_df.columns,
                                index = ['Kurtosis'])

    def _calc_hit_ratio(self, return_df):
        indic_df = return_df.where(return_df > 0,0).where(return_df < 0,1)
        return pd.DataFrame([[indic_df[col].mean() for col in indic_df.columns]],
                            columns = return_df.columns,
                            index = ['HitRatio'])
            

    def _calc_aggregated_hitratio(self, return_df, is_monthly=True):
        if is_monthly:
            date_list = cf.create_monthly_datelist(start_date=return_df.index[0],
                                                   end_date=return_df.index[-1])
        else:
            date_list = cf.create_weekly_datelist(start_date=return_df.index[0],
                                                  end_date=return_df.index[-1])
        date_list = np.array(date_list)
        aggr_return = copy.deepcopy(return_df)
        aggr_return['AggregateDate'] = [date_list[date_list>=return_df.index[i]][0] 
                                      if np.any(date_list>=return_df.index[i]) \
                                      else return_df.index[-1]\
                                      for i in range(return_df.shape[0])]
        
        grouped_df = aggr_return.groupby(['AggregateDate']).sum()
        return pd.DataFrame(grouped_df.where(grouped_df>0).count()/ grouped_df.shape[0]).T
