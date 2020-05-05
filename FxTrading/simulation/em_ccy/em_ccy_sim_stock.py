# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:30:00 2020
@author: nory.xjr1
"""
import os
import logging
import numpy as np
import pandas as pd

from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from simulation.em_ccy.em_ccy_sim_fc import EMCcySim
from util.db_connector import DBConnector
import util.common_func as cf
from util.performance_measurer import PerformanceMeasurer


class EMCcySimByStock(EMCcySim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        #self._base_date_list = cf.create_weekly_datelist(self._start_date,
        #self._end_date)
        self._ir_diff = kwargs.get('ir_diff', 3)
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def simulate(self):
        self._logger.info("Simulation Starting...")
        src_date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        rate_return_df = self._calc_return(self._price_df[self._em_rate_tickers].loc[src_date_list])
        stock_diff_df = pd.DataFrame(np.array(self._price_df[self._fc_tickers].loc[src_date_list].iloc[1:]) / np.array(self._price_df[self._fc_tickers].loc[src_date_list].iloc[:-1]),
                                   index = self._price_df.loc[src_date_list].index[1:],
                                   columns = self._fc_tickers)
        src_return_df = pd.merge(rate_return_df, stock_diff_df, right_index=True, left_index=True)
        normalized_df = pd.DataFrame([[self._normalize(src_return_df[ticker], value_date) 
                                        for value_date in self._date_list[1:]] 
                                       for ticker in self._em_rate_tickers + self._fc_tickers],
                                      index = self._em_rate_tickers + self._fc_tickers, 
                                      columns = self._date_list[1:]).T.dropna(axis=0)
        if self._exp_return_file is None:
            self._logger.info("Selecting EM Currency Tickers usgin Rate")
            em_prior_tickers = pd.DataFrame([(self._em_rate_price_dic[normalized_df[self._em_rate_tickers].iloc[i].idxmax()], 
                                              self._em_rate_price_dic[normalized_df[self._em_rate_tickers].iloc[i].idxmin()])
                                             for i in range(normalized_df.shape[0])],
                                            index = normalized_df.index,
                                            columns = ['best', 'worst'])
        else:
            self._logger.info("Selecting EM Currency Tickers usgin Expected Return")
            exp_return_df = pd.read_csv(self._exp_return_file)
            exp_return_df = cf.convert_date_format(exp_return_df, target_col='ValueDate').set_index('ValueDate')
            em_prior_tickers = pd.DataFrame([(exp_return_df[self._price_tickers].iloc[i].idxmax(), 
                                              exp_return_df[self._price_tickers].iloc[i].idxmin())
                                             for i in range(exp_return_df.shape[0])],
                                            index = exp_return_df.index,
                                            columns = ['best', 'worst'])

        if self._has_indication_diff:#one week delay, like Chicago FC
            indict_df = pd.DataFrame([False] + normalized_df[self._fc_tickers[0]].iloc[:-1]\
                                                    .apply(lambda x: True if x < self._fc_threshold else False).tolist(),
                                            index = normalized_df.index,
                                            columns = ['fc_priority'])
        else:
            min_ir_return = normalized_df[self._em_rate_tickers].min(axis=1)
            ir_indication = pd.DataFrame([np.all(min_ir_return.iloc[i:i + self._ir_diff] < 0) 
                                          for i in range(min_ir_return.shape[0] - (self._ir_diff - 1))],
                                         index = min_ir_return.index[(self._ir_diff - 1):],
                                         columns = ['ir_priority'])
            #ir_indication = pd.DataFrame([min_ir_return.iloc[i:i + self._ir_diff].mean() < 0
            #                              for i in range(min_ir_return.shape[0] - (self._ir_diff - 1))],
            #                             index = min_ir_return.index[(self._ir_diff - 1):],
            #                             columns = ['ir_priority'])
            stock_indication = pd.DataFrame(stock_diff_df[self._fc_tickers[0]]\
                                           .apply(lambda x: True if x > self._fc_threshold else False).tolist(),
                                            index = stock_diff_df.index,
                                            columns = ['fc_priority'])

            #indict_df = stock_indication
            #indict_df.columns = ['fc_priority']
            indict_df = pd.DataFrame(pd.merge(ir_indication,
                                              stock_indication,
                                              right_index=True, 
                                              left_index=True)\
                                           .sum(axis=1).apply(lambda x:True if x == 2 else False),
                                            columns = ['fc_priority'])
            

        sign_df = pd.merge(em_prior_tickers, indict_df, right_index=True, left_index=True)
    
        self._logger.info("Building Position...")
        #Risk On: Long EM Ccy of Worst Score ->Position: -1(USD Short, EM Long)
        #of Worst
        #Risk OFF: Short EM Ccy of Best Score ->Position: 1(USD Long, EM Short)
        #of Best
        position_df = pd.DataFrame([(sign_df.iloc[i]['worst'], -1) 
                                    if sign_df.iloc[i]['fc_priority']
                                     else (sign_df.iloc[i]['best'], 1) 
                                     for i in range(sign_df.shape[0])],
                                     index = sign_df.index,
                                     columns=['ccy', 'ls'])
        position_df.index.name = 'ValueDate'
    
        #position_df = pd.DataFrame([[-1 / len(self._price_tickers)
        #                             for j in range(len(self._price_tickers))]
        #                            if sign_df.iloc[i]['fc_priority']
        #                             else [1 / len(self._price_tickers)
        #                             for j in range(len(self._price_tickers))]
        #                             for i in range(sign_df.shape[0])],
        #                             index = sign_df.index)

        if self._includes_swap:
            price_return_df = self._calc_return_inc_swap(self._price_df[self._price_tickers + [self._em_price_fwd_dic[k] 
                                                                        for k in self._em_price_fwd_dic.keys()]].loc[self._date_list],
                                                         self._price_tickers,
                                                         self._em_price_fwd_dic).loc[position_df.index]
        else:
            price_return_df = self._calc_return(self._price_df[self._price_tickers].loc[self._date_list], 
                                                with_log=True).loc[position_df.index]

        self._logger.info("Calculating Perofrmance...")
        #import pdb;pdb.set_trace()
        return_series_df = pd.DataFrame([price_return_df[position_df.iloc[i][0]].iloc[i + 1] * position_df.iloc[i][1]
                                         for i in range(position_df.shape[0] - 1)],
                                        index = position_df.index[:-1],
                                        columns=['return'])
        
        #return_series_df = pd.DataFrame((np.array(price_return_df)[1:] *
        #position_df[:-1]).sum(axis=1),
        #                                columns=['return'])
        return_series_df.index.name = 'ValueDate'
        return_series_df['cum_return'] = return_series_df['return'].cumsum()
        
        
        #output result
        output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
        self.output_detaild_result(position_df, return_series_df, output_suffix)
        pd.merge(return_series_df, sign_df, right_index=True, left_index=True)\
          .to_csv(os.path.join('output', 'em_reutrn_series_{1}_{0}.csv'.format(output_suffix, self._fc_tickers[0])))
        perform_measurer = PerformanceMeasurer()
        #perform_measurer.create_result_summary(return_series_df['return']).to_csv('em_performance.csv')
        perform_measurer.create_result_summary(return_series_df)[['return']]\
            .to_csv(os.path.join('output','em_performance_{1}_{0}.csv'.format(output_suffix, self._fc_tickers[0])))

        self._logger.info("Simulation Complated.")


if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    roll_term = 104
    #start_date = date(2003, 3, 28)
    start_date = date(2007, 1, 1) - relativedelta(weeks=roll_term + 2)
    end_date = date.today()#date(2019, 12, 27)

    price_tickers = ['USDZAR Index', 'USDMXN Index']
    rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index']
    fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy']

    #price_tickers = ['USDZAR Index', 'USDMXN Index', 'USDTRY Index']
    #rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index', 'GTRU2YR Index']
    #fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy', 'USDTRY1W BGN
    #Curncy']
    import os
    fc_ticker_list = ['SPX Index']
    #fc_ticker_list = ['MXEF Index', 'NFCIINDX Index', 'HG1 Index', 'MXWO Index', 
    #                  'CFNAI Index', 'DXY Curncy', 'VIX Index',
    #                  'CTOTUSD Index', 'GSUSFCI Index', 'EURUSD Index',
    #                  'SPX Index', 'INDU Index']
    exp_return_list = os.path.join('input', 'em_ccy_expected_return_dnn.csv')
    for fc_ticker in fc_ticker_list:
        em_ccy_sim = EMCcySimByStock(start_date=start_date, 
                                     end_date=end_date, 
                                     rolls=True,
                                     roll_term=roll_term,
                                     price_tickers=price_tickers,
                                     em_rate_tickers=rate_tickers,
                                     em_fwd_tickers=fwd_tickers,
                                     #exp_return_file=exp_return_file,
                                     #fc_tickers=['NFCIINDX Index'],
                                     fc_tickers=[fc_ticker],
                                     #fc_tickers=['CTOTUSD Index'],
                                     fc_threshold=1.001,
                                     has_indication_diff=False,
                                     ir_diff=1,)
        em_ccy_sim.simulate()
