# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date

import util

def create_ccy_list():
    return ['EURUSD Index',
            'USDJPY Index',
            'EURJPY Index',
            'GBPJPY Index',
            'AUDJPY Index',
            'NZDJPY Index',
            'CHFJPY Index',
            ]


def get_fx_rate(ccy_list, 
                start_date=date(1999, 1, 1), 
                end_date=date(2019, 7, 5)):
    rate_query = "SELECT ValueDate, Ticker, Last \
                    FROM bbg_marketprice \
                   WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                     AND Ticker IN ('{2}') "\
                 .format(start_date, end_date, "','".join(ccy_list))
    with Util.DBConnector(DBName='marketdb') as db_conn:
        return db_conn.getData(rate_query).pivot(index='ValueDate',
                                                 columns='Ticker',
                                                 values='Last')\
                                          .fillna(method='ffill')

def exec_adf_test(fx_rate_df, term=30):
    pvalue_marix = []
    
    for ccy in fx_rate_df.columns:
        print("testing {ccy}...".format(ccy=ccy))
        pvalue_list = []
        for i in range(term, fx_rate_df.shape[0]):
            target_rate = fx_rate_df[ccy].iloc[i-term:i,]
            #import pdb; pdb.set_trace()
            pvalue_list.append(sm.tsa.stattools.adfuller(target_rate, regression='ct')[1])

        pvalue_marix.append(pvalue_list)

    
    return pd.DataFrame(pvalue_marix, index = fx_rate_df.columns, columns = fx_rate_df.index[term:]).T

if __name__ == '__main__':
    ccy_list = create_ccy_list()
    term_list = [14, 21, 35, 42, 63, 130, 261, 522]
    term = 14
    for  term in term_list:
        exec_adf_test(get_fx_rate(ccy_list), term=term).to_csv('adf_test_{term}days.csv'.format(term=term))
