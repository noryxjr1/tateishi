import numpy as np
import pandas as pd
from datetime import date
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from itertools import combinations
import multiprocessing as mp
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

def make_comb_list(ccy_list, min_elements=4):
    comb_list = []
    for i in range(min_elements, len(ccy_list)):
    #for i in range(len(ccy_list)-1, len(ccy_list)):
        comb_list += combinations(ccy_list, i)

    return comb_list


def search_best_coint_vec(arg_dic):
    fx_df = arg_dic['fx_df']
    term = arg_dic['term']
    comb = arg_dic['comb']
    ar_diff = arg_dic['ar_diff']
    order = arg_dic.get('order', 0)
    if order == -1:
        reg = 'nc'
    elif order == 0:
        reg = 'c'
    elif oder == 1:
        reg = 'ct'
    else:
        raise Except('invalidd order argument')

    print("Processing {0}...".format(",".join(comb)))

    pvalue_list = []
    for i in tqdm(range(term, fx_df.shape[0])):
        min_pvalue = 1.0
        eigen_vec = coint_johansen(endog=fx_df[list(comb)].iloc[i - term:i], 
                                    det_order=order, 
                                    k_ar_diff=ar_diff).evec

        for j in range(len(eigen_vec)):
            try:
                pvalue = sm.tsa.stattools.adfuller((fx_df[list(comb)].iloc[i - term:i] * eigen_vec[j]).sum(axis=1),
                                                    regression=reg)[1]
            except:
                pvalue = 1.0
            if min_pvalue >= pvalue:
                min_pvalue = pvalue
            
        pvalue_list.append(min_pvalue)

    #import pdb;pdb.set_trace()
    return pd.DataFrame(pvalue_list, columns=[",".join(comb)], index=fx_df.index[term:])
    #return [",".join(comb), np.mean(pvalue_list)]

if __name__ == '__main__':
    ccy_list = ['USDJPY Index', 'EURJPY Index', 'AUDJPY Index', 'GBPJPY Index', 
                'CADJPY Index', 'CHFJPY Index', 'NZDJPY Index']#, 'SGDJPY Index',
                #'ZARJPY Index', 'TRYJPY Index', 'MXNJPY Index']
                                                                               #'SEKJPY Index', 'DKKJPY Index', 'NOKJPY Index',
                
    start_date = date(2001,1,1)
    end_date = date.today()
    date_list = cf.create_weekly_datelist(start_date, end_date)
    fx_df = get_fx_rate(ccy_list=ccy_list, start_date=start_date, end_date=end_date)
    fx_df = fx_df.apply(lambda x:np.log(x)).loc[date_list]
    #fx_df['ZARJPY Index'] = fx_df['ZARJPY Index'] * 10
    #term = 783
    term = 156
    ar_diff = 5
    #ar_diff = 3
    #coint_df = pd.DataFrame()
    pvalue_list = []
    comb_list = make_comb_list(ccy_list)

    result_df = pd.DataFrame()
    arg_list = []
    for i, comb in enumerate(comb_list):
        print("Processing {0}/{1}...".format(i, len(comb_list)))
        arg_dic = {}
        arg_dic['fx_df'] = fx_df
        arg_dic['term'] = term
        arg_dic['comb'] = comb
        arg_dic['ar_diff'] = ar_diff
        arg_dic['order'] = 0
        arg_list.append(arg_dic)
        #result_df = result_df.append([search_best_coint_vec(arg_dic)])

        if result_df.shape[0] == 0:
            result_df = search_best_coint_vec(arg_dic)
        else:
            coint_df = search_best_coint_vec(arg_dic)
            #import pdb;pdb.set_trace()
            result_df = pd.merge(result_df, coint_df, right_index=True, left_index=True)
        

    #pool = mp.Pool(mp.cpu_count()-2)
    #result_df = pd.DataFrame(pool.map(search_best_coint_vec, arg_list))

    #result_df.columns = ['CcyPair', 'P-Value']
    from datetime import datetime
    result_df.to_csv('coint_search_result_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')), index=True)
    