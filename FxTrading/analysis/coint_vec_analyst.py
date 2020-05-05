# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:34:00 2019
@author: jpbank.quants
"""
import logging
import logging.config

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from itertools import combinations
import multiprocessing as mp
from tqdm import tqdm

from util.db_connector import DBConnector
import util.common_func as cf


class CointVecAnalyst(object):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._start_date = kwargs.get('start_date', date(2001, 1, 1))
        self._end_date = kwargs.get('end_date', date.today())
        self._ar_diff = kwargs.get('ar_diff', 3)
        self._frequency = kwargs.get('frequency', 'weekly')
        if self._frequency == 'daily':
            self._date_list = cf.create_daily_datelist(self._start_date, self._end_date)
            self._term = int(kwargs.get('term_year', 3) * 261)
        elif self._frequency == 'weekly':
            self._date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
            self._term = int(kwargs.get('term_year', 3) * 52)
        else:
            self._date_list = cf.create_monthly_datelist(self._start_date, self._end_date)
            self._term = int(kwargs.get('term_year', 3) * 12)

        
        self._ccy_list = kwargs.get('ccy_list', 
                                    np.sort(['USDJPY Index', 'EURJPY Index', 'AUDJPY Index', 'GBPJPY Index', 
                                     'CADJPY Index', 'CHFJPY Index', 'NZDJPY Index']).tolist())

        self._order = kwargs.get('order', 0)
        if self._order == -1:
            self._reg = 'nc'
        elif self._order == 0:
            self._reg = 'c'
        elif self._oder == 1:
            self._reg = 'ct'
        self._fx_rate_df = np.log(self._get_fx_rate())
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def _get_fx_rate(self):
        fx_query = "SELECT ValueDate, Ticker, Last "\
                    " FROM bbg_marketprice "\
                    " WHERE ValueDate BETWEEN '{0}' AND '{1}' "\
                    " AND Ticker IN ('{2}')".format(self._start_date, 
                                                    self._end_date, 
                                                    "','".join(self._ccy_list))

        with DBConnector(DBName='marketdb') as db_conn:
            return db_conn.get_data(fx_query).pivot(index='ValueDate', 
                                                    columns='Ticker', 
                                                    values='Last')\
                                             .loc[self._date_list]

    def _make_comb_list(self, min_elements=3):
        comb_list = []
        for i in range(min_elements, len(self._ccy_list)):
            comb_list += combinations(self._ccy_list, i)

        return comb_list

    def get_target_cointvec(self):
        pvalue_df, weight_df = self.calc_coint_vec()
        weight_df.to_csv('raw_weight.csv')
        result_df = pd.DataFrame()
        for value_date in pvalue_df.index:
            target_ccy = pvalue_df.query("index == @value_date").T.idxmin()[0]
            coint_df = pd.DataFrame(weight_df.query("index == @value_date")).query("Portfolio == @target_ccy")
            
            one_df=pd.DataFrame([np.ones(len(self._ccy_list))], 
                                index=['Weight'], 
                                columns=self._ccy_list)
            #import pdb;pdb.set_trace()
            result_df = result_df.append((one_df * coint_df.reset_index('ValueDate')\
                                                           .set_index('Ccy')[['Weight']].astype(float).T)\
                                                           .fillna(0.0))
        
        result_df.index = pvalue_df.index

        return result_df, pvalue_df


    def calc_coint_vec(self):
        value_list = []
        comb_list = self._make_comb_list()

        pvalue_df = pd.DataFrame()
        weight_df = pd.DataFrame()
        for i, comb in enumerate(comb_list):
            self._logger.info("Processing {0}/{1}...".format(i, len(comb)))
            
            if pvalue_df.shape[0] == 0:
                pvalue_df, weight_df = self._search_best_coint_vec(comb)
            else:
                lhs, rhs = self._search_best_coint_vec(comb)
                pvalue_df = pd.merge(pvalue_df, lhs, 
                                     right_index=True, left_index=True)
                weight_df = weight_df.append(rhs)
        return pvalue_df, weight_df


    def _search_best_coint_vec(self, comb):

        self._logger.info("Processing {0}...".format(",".join(comb)))
        #comb = ('EURJPY Index','GBPJPY Index', 'CHFJPY Index', 'AUDJPY Index', 'NZDJPY Index')
        comb = np.sort(comb).tolist()
        weight_df = pd.DataFrame()
        pvalue_list = []
        for i in tqdm(range(self._term, self._fx_rate_df.shape[0])):
            value_date = self._fx_rate_df.index[i]
            start_date = self._fx_rate_df.index[i-self._term]
            #value_date = date(2019,1,4)
            #start_date = value_date - relativedelta(weeks=self._term)
            target_fx = self._fx_rate_df[list(comb)].query("index>@start_date & index<=@value_date")
            min_pvalue = 1.0
            target_vec = []
            eigen_vec = coint_johansen(#endog=self._fx_rate_df[list(comb)].query("index>@start_date & index<=@value_date"),#.iloc[i - self._term:i], 
                                        endog=target_fx,
                                        det_order=self._order, 
                                        k_ar_diff=self._ar_diff).evec

            for j in range(len(eigen_vec)):
                try:
                    pvalue = sm.tsa.stattools.adfuller((target_fx*eigen_vec[j]).sum(axis=1),
                                                       #(self._fx_rate_df[list(comb)].iloc[i - self._term:i] * eigen_vec[j]).sum(axis=1),
                                                       #(self._fx_rate_df[list(comb)].query("index>@start_date & index<=@value_date") * eigen_vec[j]).sum(axis=1),
                                                       regression=self._reg)[1]
                except:
                    pvalue = 1.0
                if min_pvalue >= pvalue:
                    min_pvalue = pvalue
                    target_vec = eigen_vec[j]
            
            pvalue_list.append(min_pvalue)
            #import pdb;pdb.set_trace()
            weight_df = weight_df.append(pd.DataFrame(np.array([np.repeat(','.join(comb), len(target_vec)),
                                                                comb,
                                                                target_vec]).T, 
                                                      index=np.repeat(value_date, len(target_vec)),
                                                      columns=['Portfolio', 'Ccy', 'Weight']))
        
        weight_df.index.name='ValueDate'
        pvalue_df = pd.DataFrame(pvalue_list, columns=[",".join(comb)], 
                                 index=self._fx_rate_df.index[self._term:])
        return pvalue_df, weight_df

if __name__ == '__main__':
    logging.config.fileConfig('./logger_config.ini')
    from datetime import datetime
    #coint_vec_analyst = CointVecAnalyst(start_date=date(2018,1,1), term_year=1)
    coint_vec_analyst = CointVecAnalyst(term_year=2)
    weight_df, pvalue_df = coint_vec_analyst.get_target_cointvec()
    weight_df.to_csv('coint_vec_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')), index=True)
    pvalue_df.to_csv('pvalue_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')), index=True)
