import numpy as np
import pandas as pd
from datetime import date
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from tqdm import tqdm


from util.db_connector import DBConnector
import util.common_func as cf

def get_fx_rate(start_date=date(2001,1,1), 
                end_date=date.today(), 
                ccy_list=['USDJPY Index','EURJPY Index']):
    fx_query = "SELECT ValueDate, Ticker, Last "\
                " FROM bbg_marketprice "\
                " WHERE ValueDate BETWEEN '{0}' AND '{1}' "\
                " AND Ticker IN ('{2}')".format(start_date, end_date, "','".join(ccy_list))

    with DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(fx_query).pivot(index='ValueDate', columns='Ticker', values='Last')

if __name__== '__main__':
    ccy_list = ['USDJPY Index','EURJPY Index', 'AUDJPY Index',  'GBPJPY Index', 'CHFJPY Index', 'NZDJPY Index']#, 'ZARJPY Index', 'TRYJPY Index']
    start_date = date(2000,1,1)
    end_date= date.today()
    date_list = cf.create_weekly_datelist(start_date, end_date)
    fx_df = get_fx_rate(ccy_list=ccy_list, start_date=start_date, end_date=end_date).loc[date_list]
    #fx_df = fx_df.apply(lambda x:np.log(x))
    #fx_df['ZARJPY Index'] = fx_df['ZARJPY Index'] * 10
    fx_df.to_csv('fx_df.csv')
    #term = 783
    term = 156
    ar_diff = 5
    coint_df = pd.DataFrame()
    pvalue_list = []
    for i in tqdm(range(term, fx_df.shape[0])):
        min_pvalue=1.0
        target_vec = []
        eigen_vec = coint_johansen(endog=fx_df[ccy_list].iloc[i-term:i], 
                                   det_order=1, 
                                   k_ar_diff=ar_diff).evec
        for j in range(len(eigen_vec)):
            pvalue = sm.tsa.stattools.adfuller((fx_df[ccy_list].iloc[i-term:i]*eigen_vec[j]).sum(axis=1),
                                               regression='ct')[1]
            if min_pvalue >= pvalue:
                min_pvalue = pvalue
                target_vec = eigen_vec[j]
                #print(j, "selected", 'pvalue', pvalue)
            
        print("minimum p value is {0}".format(min_pvalue))
        pvalue_list.append(min_pvalue)
        print(fx_df.index[i])
        coint_df = coint_df.append([target_vec])

    coint_df.columns = ccy_list
    coint_df.index = fx_df.index[term:]
    coint_df.to_csv('coint_vec.csv')
    pd.DataFrame(pvalue_list,index=coint_df.index).to_csv('adf_pvalue_{0}.csv'.format(len(ccy_list)))
