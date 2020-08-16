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
import copy

import util.common_func as cf
from util.ml_config_parser import MLConfigParser
from simulation.em_ccy_sim_fc import EMCcySim
from util.performance_measurer import PerformanceMeasurer

class EstimatedFCSim(object):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._config = MLConfigParser()
        self._roll_term = kwargs.get('roll_term', 104)
        self._start_date = kwargs.get('start_date', date(2006, 4, 7) - relativedelta(weeks=self._roll_term + 2))
        self._end_date = kwargs.get('end_date', date.today())
        self._fc_ticker_list = kwargs.get('fc_ticker_list', ['NFCIINDX Index', 'GSUSFCI Index'])
        self._price_tickers = kwargs.get('price_tickers', ['USDZAR Index', 'USDMXN Index'])
        self._rate_tickers = kwargs.get('rate_tickers', ['GSAB2YR Index', 'GMXN02YR Index'])
        self._fwd_tickers = kwargs.get('fwd_tickers', ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy'])

        self._all_fc_label = pd.read_csv(kwargs.get('fc_label',
                                                   os.path.join(self._config.input_dir, 
                                                                self._config.fc_label_file)))
        self._alg_list = np.unique(self._all_fc_label.Algorithm).tolist()
        
        #ToDo: set value from config file or argument
        self._threshold_dic = {}
        self._threshold_dic['GSUSFCI Index'] = {'Upper': 0.6, 'Lower': -0.6}
        self._threshold_dic['NFCIINDX Index'] = {'Upper': 3, 'Lower': -3}

        self._return_dic = {}
        self._all_return_df = pd.DataFrame()
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    @property
    def return_series(self):
        return self._all_return_df


    def execute_normal_fc_sim(self):
        self._logger.info("Executing Normal Financial Condition Simulation...")
        return_df = pd.DataFrame()
        normalized_df = pd.DataFrame()
        for fc_ticker in self._fc_ticker_list:
            self._logger.info("Processing {0}...".format(fc_ticker))
           
            em_ccy_sim = EMCcySim(start_date=self._start_date, 
                                  end_date=self._end_date, 
                                  rolls=True,
                                  roll_term=self._roll_term,
                                  price_tickers=self._price_tickers,
                                  em_rate_tickers=self._rate_tickers,
                                  em_fwd_tickers=self._fwd_tickers,
                                  fc_tickers = [fc_ticker],
                                  fc_threshold = -0.05 if fc_ticker == 'GSUSFCI Index' else 0,
                                  has_indication_diff=False if fc_ticker == 'GSUSFCI Index' else True
                                  )
            em_ccy_sim.simulate()
            if return_df.shape[0] == 0:
                return_df = em_ccy_sim.return_series[['return']]
            else:
                return_df = pd.merge(return_df, em_ccy_sim.return_series[['return']],
                                     right_index = True, left_index = True)

            if normalized_df.shape[0] == 0:
                normalized_df = em_ccy_sim.fc_normalized.shift(1)
            else:
                normalized_df = pd.merge(normalized_df, em_ccy_sim.fc_normalized,
                                         right_index = True, left_index = True)

        return_df.columns = self._fc_ticker_list
        self._logger.info("Normal Financial Condition Simulation Completed.")
        return return_df, normalized_df


    def execute(self):
        return_df, normalized_df = self.execute_normal_fc_sim()

        self._logger.info("Calculating Return with Estimated Financial Condition...")
        return_matrix = []
        for alg in self._alg_list:
            self._logger.info("Processing {0}...".format(alg))
            fc_label = self._all_fc_label.query("Algorithm == @alg")[['ValueDate', 'Predict']]
            fc_label = cf.convert_date_format(fc_label)
            fc_label.set_index('ValueDate', inplace=True)

            return_list = []
            for value_date in return_df.index:
                if value_date in fc_label.index:
                
                    target_ticker = self._fc_ticker_list[int(fc_label.loc[value_date].Predict)]
                    if normalized_df['GSUSFCI Index'].loc[value_date] < self._threshold_dic[target_ticker]['Upper'] \
                    and normalized_df['GSUSFCI Index'].loc[value_date] > self._threshold_dic[target_ticker]['Lower']:
                        return_list.append(return_df[target_ticker].loc[value_date])
                    else:
                        return_list.append(0)
                else:
                    return_list.append(return_df[self._fc_ticker_list[0]].loc[value_date])

            return_matrix.append(return_list)
        self._all_return_df = pd.DataFrame(return_matrix, index=self._alg_list, columns = return_df.index).T
        self._logger.info("Return Calculation Completed.")

    def output_result(self):
        output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
        self._all_return_df.to_csv(os.path.join(self._config.output_dir,
                                                'all_return_series_{0}.csv'.format(output_suffix)))
        self._all_return_df.cumsum().to_csv(os.path.join(self._config.output_dir,
                                                'cum_return_series_{0}.csv'.format(output_suffix)))
        for alg in self._all_return_df.columns:
            perform_measurer = PerformanceMeasurer()
            perform_measurer.create_result_summary(self._all_return_df[[alg]])[[alg]]\
                .to_csv(os.path.join(self._config.output_dir,
                                     '{0}_em_performance_{1}.csv'.format(alg, output_suffix)))


        self._summarize_performance(output_suffix).to_csv(os.path.join(self._config.output_dir, 
                                                                       'total_performance_{0}.csv').format(output_suffix))

    def _summarize_performance(self, output_suffix):
        target_files = glob.glob(os.path.join(self._config.output_dir, 
                                              "*em_performance_{0}.csv".format(output_suffix)))
        result_df = pd.DataFrame()
        for target_file in target_files:
            alg_name = "_".join(os.path.basename(target_file).split('_')[:2])
            alg_df = pd.read_csv(target_file)
            alg_df.columns = ['Measure', 'Return']
            alg_df['Algorithm'] = alg_name

            result_df = result_df.append(alg_df)
        
        return result_df.pivot(index='Measure', columns='Algorithm', values='Return')


if __name__ == '__main__':
    logging.config.fileConfig('./logger_config.ini')
    estimated_fc_sim = EstimatedFCSim()
    estimated_fc_sim.execute()
    estimated_fc_sim.output_result()
    
    #import pickle as pkl
    #import pdb;pdb.set_trace()
    #with open('test/ans_df.pkl', 'wb') as f:
    #    pkl.dump(estimated_fc_sim.return_series, f)
