"""This class is to create new feature"""
# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import pandas as pd
from datetime import date
import copy
import talib as ta
from functools import lru_cache

import util
import util.common_func as cf


class RSIFactorMaker(object):
    def __init__(self, *args, **kwargs):
        self._start_date = kwargs.get('StartDate', date(2004,1,2))
        self._end_date = kwargs.get('EndDate', date(2019,6,28))
        self._target_ccy = kwargs.get('TargetCcy', ['USD', 'EUR'])
        self._is_weekly = kwargs.get('is_weekly', True)

        self._price_ticker = [self._target_ccy[1] + self._target_ccy[0] + ' Index']
        self._price_df = self.create_factor(self._price_ticker)
        

        self._surprise_ticker = 'CESI' + np.array(self._target_ccy).astype(object) + ' Index'
        self._datachange_ticker = 'CECIC' + np.array(self._target_ccy).astype(object) + ' Index'
        self._ctot_ticker = 'CTOT' + np.array(self._target_ccy).astype(object) + ' Index'
        self._value_ticker = 'BISB' + np.array([ccy[:2] for ccy in self._target_ccy]).astype(object) + 'N Index'
        self._carry_ticker = ['GTDEM10Y Govt', 'GT10 Govt']

    def create_feature_vector(self):
        return_df = self._create_return_df()
        #import pdb; pdb.set_trace()
        #EWMA
        feature_vec_df = self._create_rsi_ewma()
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi_ewma(rsi_param=14, param=5), 
                                  right_index=True, left_index=True)
        
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi_ewma(rsi_param=21, param=5), 
                                  right_index=True, left_index=True)
        
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi_ewma(rsi_param=35, param=5), 
                                  right_index=True, left_index=True)
        
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi_ewma(rsi_param=42, param=5), 
                                  right_index=True, left_index=True)
        
        #Normal
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi(rsi_param=7), 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi(rsi_param=14), 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi(rsi_param=21), 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi(rsi_param=35), 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi(rsi_param=42), 
                                  right_index=True, left_index=True)


        #return feature_vec_df.dropna(axis=0)
        return pd.merge(return_df, feature_vec_df, 
                        right_index=True, 
                        left_index=True).dropna(axis=0)
        
    def _create_return_df(self, target_day=None):
        if self._is_weekly:
            date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        else:
            date_list = cf.create_monthly_datelist(self._start_date, self._end_date)
        
        weekly_price = self._price_df.loc[date_list]
        if target_day is None:
            return  pd.DataFrame(np.log(np.array(weekly_price.iloc[1:]) \
                                      / np.array(weekly_price.iloc[:-1])),
                                 index = weekly_price.index[:-1],
                                 columns = ['Return'])
        else:
            target_date_list = cf.create_weekly_datelist(self._start_date, 
                                                         self._end_date, 
                                                         weeknum=target_day)
            target_price = self._price_df.loc[target_date_list]
            if target_date_list[0] > date_list[0]:
                if target_price.shape[0] < weekly_price.shape[0]:
                    weekly_price = weekly_price.iloc[:-1]
                elif target_price.shape[0] == weekly_price.shape[0]:
                    target_price = target_price.iloc[1:]
                    weekly_price = weekly_price.iloc[:-1]
            elif target_date_list[0] < date_list[0]:
                if target_price.shape[0] > weekly_price.shape[0]:
                    target_price = target_price.iloc[1:]
                elif target_price.shape[0] == weekly_price.shape[0]:
                    target_price = target_price.iloc[1:]
                    weekly_price = weekly_price.iloc[:-1]

            return  pd.DataFrame(np.log(np.array(target_price) \
                                      / np.array(weekly_price)),
                                 index = weekly_price.index,
                                 columns = ['Return'])

    #@lru_cache()
    def _create_surprise_df(self, col_name='Surprise'):
        surprise_df = self.create_factor(self._surprise_ticker)
        surprise_df[col_name] = surprise_df['CESI'+self._target_ccy[1]+' Index'] \
                              - surprise_df['CESI'+self._target_ccy[0]+' Index']

        return surprise_df[[col_name]]

    #@lru_cache()
    def _create_surprise_ewma(self, param=5):
        surprise_df = self._create_surprise_df()
        surprise_df['Surprise_EWMA'] = pd.DataFrame(np.array(surprise_df['Surprise'].iloc[1:]) \
                                                  - np.array(surprise_df['Surprise'].iloc[:-1]),
                                                    index = surprise_df.index[1:])
        return surprise_df[['Surprise_EWMA']].ewm(span=param).mean()#-surprise_df[['Surprise_EWMA']]

    #@lru_cache()
    def _create_datachange_df(self, col_name='DataChange'):
        datachange_df = self.create_factor(self._datachange_ticker)
        if 'JPY' in self._price_ticker[0]:
            datachange_df[col_name] = datachange_df['CECIC'+self._target_ccy[1]+' Index']
        else:
            datachange_df[col_name] = datachange_df['CECIC'+self._target_ccy[1]+' Index']\
                                     -datachange_df['CECIC'+self._target_ccy[0]+' Index']

        return datachange_df[[col_name]]



    #@lru_cache()
    def _create_datachange_ewma(self, param=5):
        datachange_df = self._create_datachange_df()
        datachange_df['DataChange_EWMA'] = pd.DataFrame(np.array(datachange_df['DataChange'].iloc[1:]) \
                                                     - np.array(datachange_df['DataChange'].iloc[:-1]),
                                                       index = datachange_df.index[1:])
        return datachange_df[['DataChange']].ewm(span=param).mean()# - datachange_df[['DataChange_EWMA']]


    #@lru_cache()
    def _create_ctot_df(self, col_name='CTOT'):
        ctot_df = self.create_factor(self._ctot_ticker)
        ctot_df[col_name] = ctot_df['CTOT'+self._target_ccy[1]+' Index']\
                           -ctot_df['CTOT'+self._target_ccy[0]+' Index']
        return ctot_df[[col_name]]

    #@lru_cache()
    def _create_ctot_ewma(self, param=5):
        ctot_df = self._create_ctot_df()
        ctot_df['CTOT_EWMA'] = pd.DataFrame(np.array(ctot_df['CTOT'].iloc[1:]) \
                                         - np.array(ctot_df['CTOT'].iloc[:-1]),
                                           index = ctot_df.index[1:])
        return ctot_df[['CTOT_EWMA']].ewm(span=param).mean()# - ctot_df[['CTOT_EWMA']]


    #@lru_cache()
    def _create_rsi(self, rsi_param=7, col_name='RSI'):
        return pd.DataFrame([ta.RSI(np.array(self._price_df.iloc[:, i]), rsi_param) \
                             for i in range(self._price_df.shape[1])],
                            index=[col_name+str(rsi_param)],
                            columns=self._price_df.index).T


    def _create_rsi_ewma(self, rsi_param=7, param=5):
        rsi_df = self._create_rsi(rsi_param, 'RSI_EWMA')
        return rsi_df[['RSI_EWMA'+str(rsi_param)]].ewm(span=param).mean() - rsi_df[['RSI_EWMA'+str(rsi_param)]]


    def _create_iv_df(self, vi_type='R', delta=25, vi_term='1W'):
        target_ticker = self._target_ccy[1] + self._target_ccy[0] \
                      + str(delta) + vi_type + vi_term + ' BGN Curncy'
        return self.create_factor([target_ticker])


    def _create_vi_ewma(self, param=5, vi_type='R', delta=25, vi_term='1W'):
        vi_df = self._create_iv_df(vi_type=vi_type, delta=delta, vi_term=vi_term)
        vi_df.columns = [col.replace(self._target_ccy[1] + self._target_ccy[0], '') + '_EWMA' \
                         for col in vi_df.columns]
        return vi_df.ewm(span=param).mean() - vi_df


    def _create_fwdrate_df(self, term='1W'):
        target_ticker = self._target_ccy[1] + self._target_ccy[0] + term + ' BGN Curncy'
        return self.create_factor([target_ticker])

    def _create_fwdrate_ewma(self, param=5, term='1W'):
        rate_df = self._create_fwdrate_df(term=term)
        rate_df.columns = [col.replace(self._target_ccy[1] + self._target_ccy[0], '') + '_EWMA' \
                         for col in rate_df.columns]
        return rate_df.ewm(span=param).mean() - rate_df


    def _create_value_df(self, col_name='Value'):
        value_df = self.create_factor(self._value_ticker)
        value_df[col_name] = value_df['BISB'+self._target_ccy[1][:2]+'N Index']\
                            -value_df['BISB'+self._target_ccy[0][:2]+'N Index']
        return value_df[[col_name]]

    def _create_value_ewma(self, param=5):
        value_df = self._create_value_df()
        value_df['Value_EWMA'] = pd.DataFrame(np.array(value_df['Value'].iloc[1:]) \
                                            - np.array(value_df['Value'].iloc[:-1]),
                                              index = value_df.index[1:])
        return value_df[['Value_EWMA']].ewm(span=param).mean()# - value_df[['Value_EWMA']]


    def _create_carry_df(self, col_name='Carry'):
        carry_df = self.create_factor(self._carry_ticker)
        carry_df[col_name] = carry_df[self._carry_ticker[0]]\
                            -carry_df[self._carry_ticker[1]]
        return carry_df[[col_name]]

    def _create_carry_ewma(self, param=5):
        carry_df = self._create_carry_df()
        carry_df['Carry_EWMA'] = pd.DataFrame(np.array(carry_df['Carry'].iloc[1:]) \
                                            - np.array(carry_df['Carry'].iloc[:-1]),
                                              index = carry_df.index[1:])
        return carry_df[['Carry_EWMA']].ewm(span=param).mean()


    def _create_postion_df(self, col_name='Position'):
        return self.create_factor(['IMMBENCN Index'])\
                   .rename(columns={'IMMBENCN Index':col_name})

    def _create_position_ewma(self, param=5):
        col_name = 'Position_EWMA'
        position_df = self._create_postion_df(col_name=col_name)
        return position_df[[col_name]].ewm(span=param).mean()


    def create_factor(self, ticker_list):
        factor_query = "SELECT ValueDate, Ticker, Last \
                        FROM bbg_marketprice \
                        WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                        AND Ticker IN ('{2}')"\
                       .format(self._start_date, 
                               self._end_date, 
                               "','".join(ticker_list))
        with util.DBConnector(DBName='marketdb') as db_conn:
            factor_df = db_conn.get_data(factor_query).pivot(index='ValueDate',
                                                            columns='Ticker',
                                                            values='Last')\
                                                     .fillna(method='ffill')
        date_df = pd.DataFrame(cf.create_daily_datelist(start_date=self._start_date,
                                                        end_date=self._end_date),
                               columns=['ValueDate'])
        factor_df =  pd.merge(date_df, 
                              factor_df.reset_index('ValueDate'),
                              on='ValueDate', 
                              how='left')\
                       .set_index('ValueDate')\
                       .fillna(method='ffill')\
                       .fillna(method='bfill')
        return factor_df


if __name__ == '__main__':
    is_weely = True
    factor_maker = RSIFactorMaker(TargetCcy=['JPY','USD'], is_weekly=is_weely)
    factor_maker.create_feature_vector().to_csv('rsi_feature_vector_usdjpy_{0}.csv'.format('W' if is_weely else 'M'))

