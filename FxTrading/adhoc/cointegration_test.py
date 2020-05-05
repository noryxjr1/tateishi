import numpy as np
import pandas as pd
from datetime import date
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from util.db_connector import DBConnector
def get_fx_rate(start_date=date(2001,1,1), end_date=date.today(), ccy_list=['USDJPY Index','EURJPY Index']):
    fx_query = "SELECT ValueDate, Ticker, Last \
                FROM bbg_marketprice \
                WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                AND Ticker IN ('{2}')".format(start_date, end_date, "','".join(ccy_list))

    with DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(fx_query).pivot(index='ValueDate', columns='Ticker', values='Last')

if __name__== '__main__':
    ccy_list = ['USDJPY Index','EURJPY Index', 'AUDJPY Index', 'GBPJPY Index']
    fx_df = get_fx_rate(ccy_list=ccy_list)
    #return_df = np.log(fx_df).diff()
    #import pdb;pdb.set_trace()
    #coin_result = ts.coint(fx_df.loc[:, ccy_list[0]], fx_df.loc[:,ccy_list[1:]], 
    #                       trend='ctt', return_results=True)
    
    ccy_cols = ['USDJPY Index','EURJPY Index', 'AUDJPY Index', 'GBPJPY Index']
    #fx_df = pd.read_csv('factor.csv').set_index('ValueDate')
    term = 783
    coint_df = pd.DataFrame()
    for i in range(term, fx_df.shape[0]):
        coint_df = coint_df.append([coint_johansen(endog=fx_df[ccy_cols].iloc[i-term:i], 
                                                   det_order=-1, 
                                                   k_ar_diff=2).evec[0]])
        import pdb;pdb.set_trace()

    coint_df.columns = ccy_list
    coint_df.index = fx_df.index[term:]
    coint_df.to_csv('coint.csv')
    fx_df.to_csv('fx_df.csv')
    #result=coint_johansen(endog=fx_df[['USDJPY Index_USD', 'EURJPY Index_EUR', 'AUDJPY Index_AUD']].iloc[:250], det_order=-1, k_ar_diff=2)
    
    #print(result.evec)
