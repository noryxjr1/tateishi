"""This class is to create new feature"""
# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import pandas as pd
from datetime import date
import copy
import talib as ta
from functools import lru_cache
from functools import lru_cache

import util
import util.common_func as cf


class FCFactorMaker(object):
    def __init__(self, *args, **kwargs):
        self._start_date = kwargs.get('StartDate', date(2000,1,1))
        self._end_date = kwargs.get('EndDate', date(2020, 4 , 17))
        self._target_ccy = kwargs.get('TargetCcy', ['ZAR', 'USD'])
        self._is_weekly = kwargs.get('is_weekly', True)
        self._label_tickers = kwargs.get('label_tickers', ['NFCIINDX Index', 'GSUSFCI Index'])
        threshold_list = kwargs.get('threshold', [5, 0.6])
        assert len(threshold_list) == len(self._label_tickers)
        self._threshold_dic = {}
        for i in range(len(self._label_tickers)):
            self._threshold_dic[self._label_tickers[i]] = threshold_list[i]

        self._price_ticker = [self._target_ccy[1] + self._target_ccy[0] + ' Index']
        self._price_df = self.create_factor(self._price_ticker)
        
        if self._is_weekly:
            self._date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        else:
            self._date_list = cf.create_monthly_datelist(self._start_date, self._end_date)

        self._surprise_ticker = 'CESI' + self._target_ccy[0] + ' Index'
        self._datachange_ticker = 'CECIC' + self._target_ccy[0] + ' Index'
        self._ctot_ticker = 'CTOT' + self._target_ccy[0] + ' Index'
        self._value_ticker = ['BISB' + ccy[:2] + 'N Index' for ccy in self._target_ccy]
        #self._value_ticker = 'CTTWBR' + self._target_ccy[0][:2] + 'N Index'
        self._carry_ticker_dic = {'USD':'USGG2YR Index',
                                  'ZAR':'GSAB2YR Index',
                                  'MXN':'GMXN02YR Index',
                                  'TRY':'GTRU2YR Index'}

    def create_feature_vector(self):
        
        return_df = self._create_label_df()
        

        #return_df = self._create_daily_return_df()
        #EWMA
        #feature_vec_df = pd.merge(self._create_surprise_ewma(), 
        #                          self._create_datachange_ewma(), 
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_ctot_ewma(), 
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_rsi_ewma(), 
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_vi_ewma(delta=25, vi_term='1W'),
        #                          right_index=True, left_index=True)
        ##feature_vec_df = pd.merge(feature_vec_df,
        ##                          self._create_vi_ewma(delta=25, vi_term='3W'),
        ##                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_vi_ewma(delta=10, vi_term='1W'),
        #                          right_index=True, left_index=True)
        ##feature_vec_df = pd.merge(feature_vec_df,
        ##                          self._create_vi_ewma(delta=10, vi_term='3W'),
        ##                          right_index=True, left_index=True)
        ##feature_vec_df = pd.merge(feature_vec_df,
        ##                          self._create_vi_ewma(delta=25, vi_term='1W', vi_type='B'),
        ##                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_fwdrate_ewma(term='1W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_fwdrate_ewma(term='1M'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_value_ewma(),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_carry_ewma(),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df,
        #                          self._create_position_ewma(),
        #                          right_index=True, left_index=True)

        feature_vec_df = self._create_surprise_df().loc[self._date_list]
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_surprise_df().loc[self._date_list]), 
                                  right_index=True, left_index=True)

        #feature_vec_df = self._create_surprise_df()
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_surprise_df(), 
        #                          right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_datachange_df().loc[self._date_list], 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_datachange_df().loc[self._date_list]), 
                                  right_index=True, left_index=True)

        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_ctot_df().loc[self._date_list], 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_ctot_df().loc[self._date_list]), 
                                  right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._normalize_data(self._create_datachange_df().loc[self._date_list]), 
        #                          right_index=True, left_index=True)

        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_rsi().loc[self._date_list], 
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_rsi().loc[self._date_list]), 
                                  right_index=True, left_index=True)

        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_iv_df(vi_term='1W').loc[self._date_list],
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_iv_df(vi_term='1W').loc[self._date_list]),
                                  right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_iv_df(delta=25, vi_term='1W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_iv_df(delta=25, vi_term='3W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_iv_df(delta=10, vi_term='1W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_iv_df(delta=10, vi_term='3W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_iv_df(delta=10, vi_term='3W', vi_type='B'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_fwdrate_df(term='1W'),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_fwdrate_df(term='1M'),
        #                          right_index=True, left_index=True)


        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._create_value_df(),
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._normalize_data(self._create_value_df()),
                                  #right_index=True, left_index=True)

        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._create_carry_df().loc[self._date_list],
                                  right_index=True, left_index=True)
        feature_vec_df = pd.merge(feature_vec_df, 
                                  self._normalize_data(self._create_carry_df().loc[self._date_list]),
                                  right_index=True, left_index=True)

        ##Macro Index
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self.create_macro_index('NFCIINDX Index', True).loc[self._date_list],
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._normalize_data(self.create_macro_index('NFCIINDX Index', True).loc[self._date_list]),
        #                          right_index=True, left_index=True)

        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self.create_macro_index('GSUSFCI Index').loc[self._date_list],
        #                          right_index=True, left_index=True)
        #feature_vec_df = pd.merge(feature_vec_df, 
        #                          self._normalize_data(self.create_macro_index('GSUSFCI Index').loc[self._date_list]),
        #                          right_index=True, left_index=True)
        
        #return feature_vec_df.dropna(axis=0)
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
            #if i == src_df.shape[0]-1: import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        return pd.DataFrame(norm_list, index=src_df.index[term-1:], columns=src_df.columns)

    def _create_daily_return_df(self, term=5):
        return pd.DataFrame([np.log(np.array(self._price_df.iloc[i]) \
                                  / np.array(self._price_df.iloc[i-term])) \
                            for i in range(term, self._price_df.shape[0])], 
                            index=self._price_df.index[:-term], 
                            columns=self._price_df.columns)


    def _create_label_df(self):
        if self._is_weekly:
            date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        else:
            date_list = cf.create_monthly_datelist(self._start_date, self._end_date)
        
        label_query = "SELECT ValueDate, Ticker, Last \
                       FROM bbg_marketprice \
                       WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                       AND Ticker IN ('{2}')".format(self._start_date, self._end_date, 
                                                     "','".join(self._label_tickers))

        with util.DBConnector(DBName='marketdb') as db_conn:
            label_src_df = db_conn.get_data(label_query).pivot(index='ValueDate', 
                                                               columns='Ticker', 
                                                               values='Last').dropna(axis=0)
            
            label_src_df['NFCIINDX Index'] = [np.nan] + label_src_df['NFCIINDX Index'].iloc[:-1].tolist()

        label_src_df.dropna(axis=0, inplace=True)
        #import pdb;pdb.set_trace()
        for ticker in self._label_tickers:
            label_src_df[ticker] = self._normalize_data(label_src_df[[ticker]], term=104)
            #import pdb;pdb.set_trace()
            #label_src_df[ticker] = label_src_df[ticker].apply(lambda x: x if abs(x) < self._threshold_dic[ticker] else 0)
            
        import pdb;pdb.set_trace()
        label_src_df.to_csv('fc_label_src.csv')
        label_list = []
        for i in range(label_src_df.shape[0]):
            if label_src_df.iloc[i].max() < 0:
                label = [0, 0, 1]
            elif label_src_df.iloc[i][self._label_tickers[0]] > label_src_df.iloc[i][self._label_tickers[1]]:
                label = [1, 0, 0]
            else:
                label = [0, 1, 0]
            label_list.append(label)

        return pd.DataFrame(label_list, 
                            index=label_src_df.index, 
                            columns=self._label_tickers + ['N/A']).reindex(date_list)


    #@lru_cache()
    def _create_surprise_df(self, col_name='Surprise'):
        if type(self._surprise_ticker) == list:
            surprise_df = self.create_factor(self._surprise_ticker)
        else:
            surprise_df = self.create_factor([self._surprise_ticker])
        surprise_df[col_name] = surprise_df['CESI'+self._target_ccy[0]+' Index']

        return surprise_df[[col_name]]

    ##@lru_cache()
    #def _create_surprise_ewma(self, param=5):
    #    surprise_df = self._create_surprise_df()
    #    surprise_df['Surprise_EWMA'] = pd.DataFrame(np.array(surprise_df['Surprise'].iloc[1:]) \
    #                                              - np.array(surprise_df['Surprise'].iloc[:-1]),
    #                                                index = surprise_df.index[1:])
    #    return surprise_df[['Surprise_EWMA']].ewm(span=param).mean()#-surprise_df[['Surprise_EWMA']]

    #@lru_cache()
    def _create_datachange_df(self, col_name='DataChange'):
        if type(self._surprise_ticker) == list:
            datachange_df = self.create_factor(self._datachange_ticker)
        else:
            datachange_df = self.create_factor([self._datachange_ticker])
        
        datachange_df[col_name] = datachange_df['CECIC'+self._target_ccy[0]+' Index']

        return datachange_df[[col_name]]



    #@lru_cache()
    #def _create_datachange_ewma(self, param=5):
    #    datachange_df = self._create_datachange_df()
    #    datachange_df['DataChange_EWMA'] = pd.DataFrame(np.array(datachange_df['DataChange'].iloc[1:]) \
    #                                                 - np.array(datachange_df['DataChange'].iloc[:-1]),
    #                                                   index = datachange_df.index[1:])
    #    return datachange_df[['DataChange']].ewm(span=param).mean()# - datachange_df[['DataChange_EWMA']]


    #@lru_cache()
    def _create_ctot_df(self, col_name='CTOT'):
        if type(self._surprise_ticker) == list:
            ctot_df = self.create_factor(self._ctot_ticker)
        else:
            ctot_df = self.create_factor([self._ctot_ticker])

        ctot_df[col_name] = ctot_df['CTOT'+self._target_ccy[0]+' Index']
        return ctot_df[[col_name]]

    #@lru_cache()
    #def _create_ctot_ewma(self, param=5):
    #    ctot_df = self._create_ctot_df()
    #    ctot_df['CTOT_EWMA'] = pd.DataFrame(np.array(ctot_df['CTOT'].iloc[1:]) \
    #                                     - np.array(ctot_df['CTOT'].iloc[:-1]),
    #                                       index = ctot_df.index[1:])
    #    return ctot_df[['CTOT_EWMA']].ewm(span=param).mean()# - ctot_df[['CTOT_EWMA']]


    #@lru_cache()
    def _create_rsi(self, rsi_param=7, col_name='RSI'):
        return pd.DataFrame([ta.RSI(np.array(self._price_df.iloc[:, i]), rsi_param) \
                             for i in range(self._price_df.shape[1])],
                            index=[col_name],
                            columns=self._price_df.index).T


    #def _create_rsi_ewma(self, rsi_param=7, param=5):
    #    rsi_df = self._create_rsi(rsi_param, 'RSI_EWMA')
    #    return rsi_df[['RSI_EWMA']].ewm(span=param).mean() - rsi_df[['RSI_EWMA']]


    def _create_iv_df(self, vi_type='V', vi_term='1W'):
        target_ticker = self._target_ccy[1] + self._target_ccy[0] \
                      + vi_type + vi_term + ' BGN Curncy'
        return self.create_factor([target_ticker])


    #def _create_vi_ewma(self, param=5, vi_type='R', delta=25, vi_term='1W'):
    #    vi_df = self._create_iv_df(vi_type=vi_type, delta=delta, vi_term=vi_term)
    #    vi_df.columns = [col.replace(self._target_ccy[1] + self._target_ccy[0], '') + '_EWMA' \
    #                     for col in vi_df.columns]
    #    return vi_df.ewm(span=param).mean() - vi_df


    #def _create_fwdrate_df(self, term='1W'):
    #    target_ticker = self._target_ccy[1] + self._target_ccy[0] + term + ' BGN Curncy'
    #    return self.create_factor([target_ticker])

    #def _create_fwdrate_ewma(self, param=5, term='1W'):
    #    rate_df = self._create_fwdrate_df(term=term)
    #    rate_df.columns = [col.replace(self._target_ccy[1] + self._target_ccy[0], '') + '_EWMA' \
    #                     for col in rate_df.columns]
    #    return rate_df.ewm(span=param).mean() - rate_df


    def _create_value_df(self, col_name='Value'):
        value_df = self.create_factor(self._value_ticker)
        value_df[col_name] = value_df['BISB'+self._target_ccy[1][:2]+'N Index']\
                            -value_df['BISB'+self._target_ccy[0][:2]+'N Index']
        return value_df[[col_name]]

    #def _create_value_ewma(self, param=5):
    #    value_df = self._create_value_df()
    #    value_df['Value_EWMA'] = pd.DataFrame(np.array(value_df['Value'].iloc[1:]) \
    #                                        - np.array(value_df['Value'].iloc[:-1]),
    #                                          index = value_df.index[1:])
    #    return value_df[['Value_EWMA']].ewm(span=param).mean()# - value_df[['Value_EWMA']]


    def _create_carry_df(self, col_name='Carry'):
        carry_ticker = [self._carry_ticker_dic[ccy] for ccy in self._target_ccy]
        carry_df = self.create_factor(carry_ticker)
        carry_df[col_name] = carry_df[self._carry_ticker_dic[self._target_ccy[0]]]\
                            -carry_df[self._carry_ticker_dic[self._target_ccy[1]]]
        return carry_df[[col_name]]

    #def _create_carry_ewma(self, param=5):
    #    carry_df = self._create_carry_df()
    #    carry_df['Carry_EWMA'] = pd.DataFrame(np.array(carry_df['Carry'].iloc[1:]) \
    #                                        - np.array(carry_df['Carry'].iloc[:-1]),
    #                                          index = carry_df.index[1:])
    #    return carry_df[['Carry_EWMA']].ewm(span=param).mean()


    #def _create_postion_df(self, col_name='Position'):
    #    return self.create_factor(['IMMBENCN Index'])\
    #               .rename(columns={'IMMBENCN Index':col_name})

    #def _create_position_ewma(self, param=5):
    #    col_name = 'Position_EWMA'
    #    position_df = self._create_postion_df(col_name=col_name)
    #    return position_df[[col_name]].ewm(span=param).mean()


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


    @lru_cache()
    def create_macro_index(self, ticker, shifts=False):
        macro_query = "SELECT ValueDate, Ticker, Last \
                        FROM bbg_marketprice \
                        WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                        AND Ticker = '{2}'"\
                       .format(self._start_date, 
                               self._end_date, 
                               ticker)
        with util.DBConnector(DBName='marketdb') as db_conn:
            macro_df = db_conn.get_data(macro_query).pivot(index='ValueDate',
                                                            columns='Ticker',
                                                            values='Last')\
                                                     .fillna(method='ffill')
        #import pdb;pdb.set_trace()
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
    is_weely = True

    factor_maker = FCFactorMaker(TargetCcy=['ZAR', 'USD'], is_weekly=is_weely)
    factor_maker._create_label_df().to_csv('fc_label.csv')


    #macro_df = pd.DataFrame()
    #ccy_list = ['ZAR','MXN']#,'TRY']
    ##ccy_list = ['MXN']
    #for ccy in ccy_list:
    #    print("Processing", ccy, "...")
    #    factor_maker = FCFactorMaker(TargetCcy=[ccy, 'USD'], is_weekly=is_weely)
    #    df = factor_maker.create_feature_vector()

    #    #factor_maker.create_feature_vector().to_csv('factor_{0}.csv'.format(ccy))

    #    df.columns = np.array(df.columns).astype(object) + '_' + ccy
    #    if macro_df.shape[0] == 0:
    #        macro_df = df
    #    else:
    #        macro_df = pd.merge(macro_df, df,
    #                             right_index=True, left_index=True)

        
    #macro_df.to_csv('factor_{0}.csv'.format('2ccy'))