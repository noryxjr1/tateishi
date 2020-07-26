# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: nory.xjr1
"""
import os
import logging.config
import numpy as np
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import glob

import util.common_func as cf
from util.ml_config_parser import MLConfigParser
from simulation.em_ccy_sim_fc import EMCcySim

def summarize_performance(output_suffix, output_dir='output'):
    target_files = glob.glob(os.path.join(output_dir, "*em_performance_{0}.csv".format(output_suffix)))
    result_df = pd.DataFrame()
    for target_file in target_files:
        alg_name = target_file.split('_')[1]
        alg_df = pd.read_csv(target_file)
        alg_df.columns = ['Measure', 'Return']
        alg_df['Algorith'] = alg_name

        result_df = result_df.append(alg_df)

    return result_df.pivot(index='Measure', columns='Algorith', values='Return')


if __name__ == '__main__':
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    config = MLConfigParser()
    roll_term = 104
    start_date = date(2006, 4, 7) - relativedelta(weeks=roll_term + 2)
    #start_date = date(2007, 1, 1) - relativedelta(weeks=roll_term + 2)
    end_date = date.today()

    price_tickers = ['USDZAR Index', 'USDMXN Index']
    rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index']
    fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy']

    all_fc_label = pd.read_csv(os.path.join(config.input_dir, 
                                            config.fc_label_file))
    alg_list = np.unique(all_fc_label.Algorithm)
    output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
    for alg in alg_list:
        logger.info("Processing {0}...".format(alg))

        fc_label = all_fc_label.query("Algorithm == @alg")[['ValueDate', 'Predict']]
        fc_label.rename(columns={'Predict':'Label'}, inplace=True)
        fc_label = cf.convert_date_format(fc_label)
        fc_label.set_index('ValueDate', inplace=True)

        em_ccy_sim = EMCcySim(start_date=start_date, end_date=end_date, 
                              rolls=True,
                              roll_term=roll_term,
                              price_tickers=price_tickers,
                              em_rate_tickers=rate_tickers,
                              em_fwd_tickers=fwd_tickers,
                              use_estimated_sign=True,
                              fc_label=fc_label
                              )
        em_ccy_sim.simulate()
        em_ccy_sim.output(output_prefix=alg, output_suffix=output_suffix)

    #Output Performace Summary
    summarize_performance(output_suffix, 
                          config.output_dir).to_csv(os.path.join(config.output_dir, 
                                                                 'total_performance_{0}.csv').format(output_suffix))