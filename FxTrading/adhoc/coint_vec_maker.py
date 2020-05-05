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

def make_comb_list(ccy_list):
    comb_list = []
    for i in range(4, len(ccy_list)):
    #for i in range(len(ccy_list)-1, len(ccy_list)):
        comb_list += combinations(ccy_list, i)

    return comb_list


def make_coint_vec(arg_dic):
    fx_df = arg_dic['fx_df']
    term = arg_dic['term']
    comb = arg_dic['comb']
    ar_diff = arg_dic['ar_diff']
    order = arg_dic.get('order', 0)
    if order == -1:
        reg = 'nc'
    elif order == 0:
        reg = 'c'
    elif order == 1:
        reg = 'ct'
    else:
        raise Except('invalid order argument')

    print("Processing {0}...".format(",".join(comb)))
    weight_df = pd.DataFrame()
    pvalue_list = []
    for i in tqdm(range(term, fx_df.shape[0])):
        value_date = fx_df.index[i]
        
        min_pvalue=1.0
        target_vec = []
        
        eigen_vec_list = coint_johansen(endog=fx_df[list(comb)].iloc[i-term:i], 
                                    det_order=order, 
                                    k_ar_diff=ar_diff).evec

        for j in range(len(eigen_vec_list)):
            eigen_vec = eigen_vec_list[j]
            eigen_vec = [eigen_vec[i] / np.abs(eigen_vec).sum() for i in range(len(eigen_vec))]
            try:
                pvalue = sm.tsa.stattools.adfuller((fx_df[list(comb)].iloc[i-term:i]*eigen_vec).sum(axis=1),
                                                    regression=reg)[1]
            except:
                pvalue = 1.0
            if min_pvalue >= pvalue:
                min_pvalue = pvalue
                target_vec = eigen_vec
            
        pvalue_list.append(min_pvalue)
        weight_df = weight_df.append(pd.DataFrame(np.array([np.repeat(','.join(comb), len(target_vec)),
                                                            comb,
                                                            target_vec]).T, 
                                                  index=np.repeat(value_date, len(target_vec)),
                                                  columns=['Portfolio', 'Ccy', 'Weight']))
        

    weight_df.index.name='ValueDate'
    #import pdb;pdb.set_trace()
    pvalue_df = pd.DataFrame(pvalue_list, columns=[",".join(comb)], index=fx_df.index[term:])
    
    return weight_df, pvalue_df


def get_target_cointvec(pvalue_df, weight_df):
        ticker_list = ['USDJPY Index', 'EURJPY Index', 'AUDJPY Index', 'GBPJPY Index', 
                       'CADJPY Index', 'CHFJPY Index', 'NZDJPY Index']
        
        result_df = pd.DataFrame()
        for value_date in pvalue_df.index:
            target_ccy = pvalue_df.query("index == @value_date").T.idxmin()[0]
            coint_df = pd.DataFrame(weight_df.query("index == @value_date")).query("Portfolio == @target_ccy")
            
            one_df=pd.DataFrame([np.ones(len(ticker_list))], 
                                index=['Weight'], 
                                columns=ticker_list)
            #import pdb;pdb.set_trace()
            result_df = result_df.append((one_df * coint_df.reset_index('ValueDate')\
                                                           .set_index('Ccy')[['Weight']].astype(float).T)\
                                                           .fillna(0.0))
        
        result_df.index = pvalue_df.index
        return result_df

if __name__== '__main__':
    ccy_list = ['USDJPY Index', 'EURJPY Index', 'AUDJPY Index', 'GBPJPY Index', 
                'CADJPY Index', 'CHFJPY Index', 'NZDJPY Index']#, 'SGDJPY Index',
                #'ZARJPY Index', 'TRYJPY Index', 'MXNJPY Index']
                #'SEKJPY Index', 'DKKJPY Index', 'NOKJPY Index',
                
    start_date = date(2001,1,1)
    end_date= date.today()
    date_list = cf.create_weekly_datelist(start_date, end_date)
    fx_df = get_fx_rate(ccy_list=ccy_list, start_date=start_date, end_date=end_date)
    fx_df = fx_df.apply(lambda x:np.log(x)).loc[date_list]
    #term = 783
    term = 156
    ar_diff = 5
    #ar_diff = 3
    pvalue_list = []
    comb_list = make_comb_list(ccy_list)

    coint_vec_df = pd.DataFrame()
    pvalue_matrix = pd.DataFrame()
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
        weight_df, pvalue_df = make_coint_vec(arg_dic)
        coint_vec_df = coint_vec_df.append(weight_df)

        if pvalue_matrix.shape[0] == 0:
            pvalue_matrix = pvalue_df
        else:
            pvalue_matrix = pd.merge(pvalue_matrix, pvalue_df, 
                                     left_index=True, right_index=True)
        
    #pool = mp.Pool(mp.cpu_count()-2)
    #result_df = pd.DataFrame(pool.map(search_best_coint_vec, arg_list))

    from datetime import datetime
    coint_vec_df.to_csv('coint_weight_result_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')), index=True)
    pvalue_matrix.to_csv('pvalue_matrix_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')), index=True)
    get_target_cointvec(pvalue_matrix, coint_vec_df).to_csv('coint_vec_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')))