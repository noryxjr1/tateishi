# -*- coding: utf-8 -*-

"""
Created on Mon Feb 18 14:30:00 2020
@author: nory.xjr1
"""
import os
import logging
import numpy as np
import pandas as pd

from datetime import date
import scipy.optimize as optimize

class CcyPortfolioManager(object):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._value_date = kwargs.get('value_date', date.today())
        self._ccy_iv_dic = {'USDZAR Index':'',
                            'USDMXN Index':'',
                            'USDTRY Index':''}
        self._hv_term = kwargs.get('hv_term', None)
        self._expected_return = kwargs.get('expected_return', None)
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def calc_hist_vol(self, price_df):
        #import pdb;pdb.set_trace()
        #return np.log(price_df).diff().dropna(axis=0).std(ddof=1)
        return np.log(price_df).diff().dropna(axis=0).cov()


    
    def optimize(self, price_df):
        vol_matrix = self.calc_hist_vol(price_df)

        def calc_min_vol(para):
            #import pdb;pdb.set_trace()
            return np.array(para).T.dot(vol_matrix).dot(np.array(para))

        def calc_max_sharpe():
            pass

        def constraint(x):
            return np.sum(x) - 1.0

        para = [1/price_df.shape[1] for i in range(price_df.shape[1])]
        bounds = [(0, 1) for i in range(price_df.shape[1])]
        if self._expected_return is None:#MinVol
            return optimize.fmin_slsqp(calc_min_vol, x0=para, bounds=bounds,
                                       eqcons=[constraint, ], iter=1000)
        else:#Max Sharpe
            pass

if __name__ == '__main__':
    from util.db_connector import DBConnector
    from dateutil.relativedelta import relativedelta
    value_date = date(2019, 12, 27)
    start_date = value_date - relativedelta(years=1)
    with DBConnector(DBName='marketdb') as db_conn:
        price_query = "SELECT ValueDate, Ticker, Last \
                       FROM bbg_marketprice \
                       WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                       AND Ticker IN ('USDZAR Index', 'USDMXN Index')".format(start_date, value_date)
        price_df = db_conn.get_data(price_query).pivot(index='ValueDate', columns='Ticker', values='Last')
    ccy_port_mgr = CcyPortfolioManager(value_date = value_date)
    print(ccy_port_mgr.optimize(price_df))



