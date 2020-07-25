"""This class is to create new feature"""
# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import copy
from functools import lru_cache

import util
import util.common_func as cf
from simulation.em_ccy_sim_fc import EMCcySim

class FCFactorMaker(object):
    def __init__(self, *args, **kwargs):
        self._input_data = cf.convert_date_format(pd.read_csv(os.path.join(os.path.dirname(__file__), '../input', 'all_input_data.csv')))

        self._start_date = kwargs.get('StartDate', date(2000,1,1))
        self._end_date = kwargs.get('EndDate', date(2020, 7, 17))
        self._target_ccy = kwargs.get('TargetCcy', ['ZAR', 'MXN'])
        self._base_ccy = kwargs.get('BaseCcy', 'USD')
        self._label_tickers = kwargs.get('label_tickers', 
                                         ['NFCIINDX Index', 'GSUSFCI Index'])
        self._ticker_threshold = kwargs.get('ticker_threshold', 
                                            {'GSUSFCI Index':-0.05, 'NFCIINDX Index':0})
        self._threshold_dic = {}
        for i in range(len(self._label_tickers)):
            if self._label_tickers[i] == 'GSUSFCI Index':
                self._threshold_dic[self._label_tickers[i]] = {'Upper': 0.6, 'Lower': -0.6}
            else:
                self._threshold_dic[self._label_tickers[i]] = {'Upper': 3, 'Lower': -3}

        self._price_ticker = [self._base_ccy + self._target_ccy[0] + ' Index',
                              self._base_ccy + self._target_ccy[1] + ' Index']
        self._price_df = self.create_factor(self._price_ticker)
        
        self._date_list = cf.create_weekly_datelist(self._start_date, self._end_date)

        self._surprise_ticker = 'CESI' + self._target_ccy[0] + ' Index'
        self._datachange_ticker = 'CECIC' + self._target_ccy[0] + ' Index'
        self._ctot_ticker = 'CTOT' + self._target_ccy[0] + ' Index'
        self._value_ticker = ['BISB' + ccy[:2] + 'N Index' for ccy in self._target_ccy]
        self._fc_tickers = ['NFCIINDX Index', 'GSUSFCI Index']
        
        self._carry_ticker_dic = {'USD':'USGG2YR Index',
                                  'ZAR':'GSAB2YR Index',
                                  'MXN':'GMXN02YR Index',
                                  'TRY':'GTRU2YR Index'}

    def create_feature_vector(self):
        deviation_df = self._create_price_deviation_df()
        return_df, normalized_df, price_return_df = self._create_label_df()
        carry_df, rate_diff = self._create_carry_df(is_return=True, col_name='Carry')
        carry_df = carry_df.reindex(self._date_list)
        rate_diff = rate_diff.reindex(self._date_list)

        feature_vec_df = pd.merge(carry_df,
                                  normalized_df,
                                  right_index=True, left_index=True)
            
        feature_vec_df = pd.merge(feature_vec_df,
                                  rate_diff,
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df,
                                  price_return_df,
                                  right_index=True, left_index=True)
        
        feature_vec_df = pd.merge(feature_vec_df,
                                  self._create_price_deviation_df(),
                                  right_index=True, left_index=True)
        
        return pd.merge(return_df, feature_vec_df, 
                        right_index=True, 
                        left_index=True).dropna(axis=0)
        
    def _normalize_data(self, src_df, term=52):
        assert src_df.shape[1] == 1
        norm_list = []
        
        for i in range(term, src_df.shape[0]+1):
            mean = src_df.iloc[i-term:i].mean()
            std = src_df.iloc[i-term:i].std(ddof=0)
            norm_list.append((src_df.iloc[i-1, 0]-mean)/std)
            
        return pd.DataFrame(norm_list, index=src_df.index[term-1:], columns=src_df.columns)

    def _create_daily_return_df(self, term=5):
        return pd.DataFrame([np.log(np.array(self._price_df.iloc[i]) \
                                  / np.array(self._price_df.iloc[i-term])) \
                            for i in range(term, self._price_df.shape[0])], 
                            index=self._price_df.index[:-term], 
                            columns=self._price_df.columns)


    def _get_sim_return(self, ticker:str, has_indication_diff:bool):
        roll_term = 104
        price_tickers = ['USDZAR Index', 'USDMXN Index']
        rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index']
        fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy']
        em_ccy_sim = EMCcySim(start_date=self._start_date,
                              end_date=self._end_date, 
                              rolls=True,
                              roll_term=roll_term,
                              fc_tickers=[ticker],
                              price_tickers=price_tickers,
                              em_rate_tickers=rate_tickers,
                              em_fwd_tickers=fwd_tickers,
                              fc_threshold=self._ticker_threshold[ticker],
                              has_indication_diff=has_indication_diff)
        em_ccy_sim.simulate()
        
        fc_normalized_df = em_ccy_sim.fc_normalized
        if has_indication_diff:
            fc_normalized_df = fc_normalized_df.shift()

        return em_ccy_sim.return_series[['return']], fc_normalized_df, em_ccy_sim.price_return


    def _create_label_df(self):
        date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        
        sim_return_df = pd.DataFrame()
        fc_normalized_df = pd.DataFrame()
        price_return_df = pd.DataFrame()
        for lable_ticker in self._label_tickers:
            if lable_ticker == 'NFCIINDX Index':
                indic_diff = True
            else:
                indic_diff = False
            if sim_return_df.shape[0] == 0:
                sim_return_df, fc_normalized_df, price_return_df = self._get_sim_return(lable_ticker, indic_diff)
            else:
                return_df, fc_df, pr_df = self._get_sim_return(lable_ticker, indic_diff)
                sim_return_df = pd.merge(sim_return_df, return_df,
                                         right_index = True, left_index = True)
                fc_normalized_df = pd.merge(fc_normalized_df, fc_df,
                                            right_index = True, left_index = True)
                
        sim_return_df.columns = self._label_tickers
        modified_return_df = pd.DataFrame([[sim_return_df[ticker].iloc[i] 
                                                if fc_normalized_df['GSUSFCI Index'].iloc[i] < self._threshold_dic[ticker]['Upper']
                                                and fc_normalized_df['GSUSFCI Index'].iloc[i] > self._threshold_dic[ticker]['Lower'] 
                                                else 0
                                            for i in range(sim_return_df.shape[0])]
                                           for ticker in self._label_tickers],
                                          index = self._label_tickers,
                                          columns = sim_return_df.index).T

        return pd.DataFrame([0 if modified_return_df.iloc[i][self._label_tickers[0]] 
                                >= modified_return_df.iloc[i][self._label_tickers[1]] 
                                else 1
                            for i in range(modified_return_df.shape[0])], 
                            index=modified_return_df.index, 
                            columns=['Return']),\
                fc_normalized_df, price_return_df


    def _create_carry_df(self, is_return=True, col_name='Carry'):
        carry_ticker = [self._carry_ticker_dic[ccy] for ccy in self._target_ccy+[self._base_ccy]]
        carry_df = self.create_factor(carry_ticker).reindex(self._date_list)
        
        col_name_list = []
        for em_ccy in self._target_ccy:
            carry_df[em_ccy+col_name] = carry_df[self._carry_ticker_dic[em_ccy]]\
                                      - carry_df[self._carry_ticker_dic[self._base_ccy]]
            col_name_list.append(em_ccy+col_name)
        
        rate_diff_df = pd.DataFrame(carry_df.iloc[:, 0] - carry_df.iloc[:, 1],
                                    columns=['MXUSmSAUS'])

        if is_return:
            return pd.DataFrame(np.array(carry_df[col_name_list].iloc[1:]) 
                              / np.array(carry_df[col_name_list].iloc[:-1]) - 1,
                              index = carry_df.index[1:],
                              columns = col_name_list), \
                    rate_diff_df
        else:
            return carry_df[col_name_list], rate_diff_df


    def _create_price_deviation_df(self, col_name='deviation'):
        price_tickers = ['USD' + ccy + ' Index' for ccy in self._target_ccy]
        price_df = self.create_factor(price_tickers).reindex(self._date_list)
        deviation_df = pd.DataFrame()
        for ticker in price_tickers:
            dev_list = [(price_df[ticker].iloc[i-1] - price_df[ticker].iloc[i-5:i-1].mean())
                         for i in range(5, price_df.shape[0]-1)]
            
            if deviation_df.shape[0] == 0:
                deviation_df = pd.DataFrame(dev_list, 
                                            columns=[ticker[:6]+'_'+col_name], 
                                            index=price_df.index[5:-1])
            else:
                deviation_df = pd.merge(deviation_df,
                                        pd.DataFrame(dev_list, 
                                                     columns=[ticker[:6]+'_'+col_name], 
                                                     index=price_df.index[5:-1]),
                                        right_index=True, left_index=True)
        return deviation_df

    def create_factor(self, ticker_list):
        target_df = self._input_data.query("@self._start_date <= ValueDate <= @self._end_date & Ticker in @ticker_list")\
                                                     .pivot(index='ValueDate',
                                                            columns='Ticker',
                                                            values='Last')\
                                                     .fillna(method='ffill')

        date_df = pd.DataFrame(cf.create_daily_datelist(start_date=self._start_date,
                                                        end_date=self._end_date),
                               columns=['ValueDate'])
        return pd.merge(date_df, 
                              target_df.reset_index('ValueDate'),
                              on='ValueDate', 
                              how='left')\
                       .set_index('ValueDate')\
                       .fillna(method='ffill')\
                       .fillna(method='bfill')


    @lru_cache()
    def create_macro_index(self, ticker, shifts=False):
        macro_df = self._input_data.query("'{0}' <= ValueDate <= '{1}'Ticker '{2}'"
                               .format(self._start_date, 
                                       self._end_date, 
                                       ticker)).pivot(index='ValueDate',
                                                            columns='Ticker',
                                                            values='Last')\
                                                     .fillna(method='ffill')
        if shifts: macro_df = macro_df.shift(1).dropna(axis=0)
        date_df = pd.DataFrame(cf.create_daily_datelist(start_date=self._start_date,
                                                        end_date=self._end_date),
                               columns=['ValueDate'])
        macro_df =  pd.merge(date_df, 
                              macro_df.reset_index('ValueDate'),
                              on='ValueDate', 
                              how='left')\
                       .set_index('ValueDate')\
                       .fillna(method='ffill')\
                       .fillna(method='bfill')
        return macro_df

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('./logger_config.ini')
    is_weely = True

    factor_maker = FCFactorMaker(TargetCcy=['ZAR', 'MXN'], is_weekly=is_weely)
    factor_maker.create_feature_vector().to_csv('fc_feature.csv')
