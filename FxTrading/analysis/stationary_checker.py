# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:34:00 2019
@author: jpbank.quants
"""
import logging
import logging.config

import numpy as np
import pandas  as pd
import statsmodels.api as sm
from datetime import datetime,date
from dateutil.relativedelta import relativedelta
#from analysis.coint_vec_analyst import CointVecAnalyst
import util.common_func as cf

if __name__ == '__main__':
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")

    import_file_name = './input/coint_vec_2y.csv'
    coint_vec_df = pd.read_csv(import_file_name)
    coint_vec_df['ValueDate'] = cf.convert_date_format(coint_vec_df)
    coint_vec_df.set_index('ValueDate', inplace=True)
    ccy_list = coint_vec_df.columns.tolist()
    term_week = 104

    fx_rate_df = np.log(cf.get_fx_rate(start_date = coint_vec_df.index[0] - relativedelta(weeks=term_week),
                                       end_date = coint_vec_df.index[-1],
                                       ccy_list = ccy_list))
    output_df = pd.DataFrame()
    price_list = []
    for i in range(coint_vec_df.shape[0]-1):
        value_date = coint_vec_df.index[i]
        #value_date = date(2019,2,22)
        logger.info("Processing in {0}".format(value_date))
        start_date = value_date - relativedelta(weeks=term_week)
        next_date = coint_vec_df.index[i+1]
        normal_date_list = cf.create_weekly_datelist(start_date, value_date)
        follow_date_list = cf.create_weekly_datelist(start_date, next_date)

        weight_df = coint_vec_df.iloc[i]
        #weight_df = coint_vec_df.loc[value_date]
        follow_price_df = fx_rate_df.loc[follow_date_list].query("index > @start_date & index <= @next_date")[ccy_list]#
        normal_price_df = fx_rate_df.loc[normal_date_list].query("index > @start_date & index <= @value_date")[ccy_list]#
        follow_pvalue = sm.tsa.stattools.adfuller((follow_price_df[ccy_list]*weight_df).sum(axis=1),
                                                 regression='c')[1]
        normal_pvalue = sm.tsa.stattools.adfuller((normal_price_df[ccy_list]*weight_df).sum(axis=1),
                                                  regression='c')[1]
        price_list.append((normal_price_df[ccy_list]*weight_df).sum(axis=1).tolist())
        #import pdb;pdb.set_trace()
        output_df = output_df.append([[value_date, normal_pvalue, follow_pvalue]])

    output_df.columns = ['ValueDate', 'Current', 'Following']
    output_df.to_csv('./output/stationary_checker_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')))
    pd.DataFrame(price_list, index=coint_vec_df.index[:-1]).to_csv('./output/index_price_{0}.csv'.format(term_week))
