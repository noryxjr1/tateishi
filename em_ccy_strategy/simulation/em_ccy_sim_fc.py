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

#from util.db_connector import DBConnector
import util.common_func as cf
from util.performance_measurer import PerformanceMeasurer


class EMCcySim(object):
    def __init__(self, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))

        self._input_data = cf.convert_date_format(pd.read_csv(os.path.join(os.path.dirname(__file__), '../input', 'all_input_data.csv')))
        self._fc_threshold = kwargs.get('fc_threshold', 0)
        self._includes_swap = kwargs.get('includes_swap', True)

        self._rolls = kwargs.get('rolls', False)
        self._start_date = kwargs.get('start_date', date(2003, 3, 28))
        self._end_date = kwargs.get('end_date', date(2020, 7, 24))

        self._use_estimated_sign = kwargs.get('use_estimated_sign', False)
        if self._use_estimated_sign:
            
            self._fc_label = kwargs.get('fc_label', None)
            assert self._fc_label is not None
        self._has_indication_diff = kwargs.get('has_indication_diff', True)

        if self._rolls:
            self._roll_term = kwargs.get('roll_term', 52)
            self._date_list = cf.create_weekly_datelist(self._start_date + relativedelta(weeks=self._roll_term), 
                                                        self._end_date)
        else:
            self._date_list = cf.create_weekly_datelist(self._start_date, self._end_date)

        self._price_tickers = kwargs.get('price_tickers',
                                         ['USDZAR Index', 'USDMXN Index'])
        self._em_rate_tickers = kwargs.get('em_rate_tickers',
                                           ['GSAB2YR Index', 'GMXN02YR Index'])
        self._em_fwd_tickers = kwargs.get('em_fwd_tickers', 
                                          ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy'])

        assert len(self._em_fwd_tickers) == len(self._em_rate_tickers) == len(self._price_tickers)

        self._em_price_rate_dic, self._em_rate_price_dic = self._create_ticker_dic(self._price_tickers, self._em_rate_tickers)
        self._em_price_fwd_dic, self._em_fwd_price_dic = self._create_ticker_dic(self._price_tickers, self._em_fwd_tickers)

        self._exp_return_file = kwargs.get('exp_return_file', None)
        self._base_rate_ticker = kwargs.get('base_rate_ticker', 'USGG2YR Index')
        self._fc_tickers = kwargs.get('fc_tickers', ['NFCIINDX Index', 'GSUSFCI Index'])
        
        self._price_df = self._get_price(self._price_tickers + self._em_rate_tickers + self._em_fwd_tickers + [self._base_rate_ticker] + self._fc_tickers)

        #calculate rate diff
        for em_ticker in self._em_rate_tickers:
            self._price_df[em_ticker] = self._price_df[em_ticker] - self._price_df[self._base_rate_ticker]


        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def return_series(self) -> pd.DataFrame:
        return self._return_series_df

    @property
    def fc_normalized(self) -> pd.DataFrame:
        return self._fc_normalized_df

    @property
    def price_return(self) -> pd.DataFrame:
        return self._price_return_df


    def _create_ticker_dic(self, price_tickers, rate_ticers):
        price_rate_dic = {}
        rate_price_dic = {}
        for price_ticker, rate_ticker in zip(price_tickers, rate_ticers):
            price_rate_dic[price_ticker] = rate_ticker
            rate_price_dic[rate_ticker] = price_ticker

        return price_rate_dic, rate_price_dic

    def _get_price(self, ticker_list):
        self._logger.info('Getting Data From Master Data...')
        return self._input_data.query("@self._start_date <= ValueDate <= @self._end_date & Ticker in @ticker_list").pivot(index='ValueDate',
                                                            columns='Ticker',
                                                            values='Last')\
                                                     .fillna(method='ffill')
    
                               
    def _normalize(self, target_df, end_date):
        if self._rolls:
            start_date = end_date - relativedelta(weeks=self._roll_term)
            df = pd.DataFrame(target_df).query("index >= @start_date & index <= @end_date")
        else:
            df = pd.DataFrame(target_df).query("index <= @end_date")
        mean = df.mean().iloc[0]
        std = df.std(ddof=0).iloc[0]
        return ((df.iloc[-1] - mean) / std).iloc[0]


    def _calc_return(self, price_df, with_log=False):
        if with_log:
            self._logger.info("Calculating Log Return...")
            return_df = np.log(price_df).diff().fillna(method='ffill').dropna(axis=0)
            return_df.index = price_df.index[:-1]
            return return_df
        else:
            self._logger.info("Calculating Normal Return...")
            return price_df.diff().dropna(axis=0) / np.array(price_df.iloc[:-1])

    def _calc_return_inc_swap(self, price_df, price_tickers, fwd_dic):
        self._logger.info("Calculating Log Return including Swap Cost...")
        return pd.DataFrame(np.log([[(price_df.iloc[i + 1][em_ticker] - price_df.iloc[i + 1][fwd_dic[em_ticker]] / 10 ** 4) / price_df.iloc[i][em_ticker] 
                              for em_ticker in price_tickers]
                             for i in range(price_df.shape[0] - 1)]),
                            index = price_df.index[1:],
                            columns = price_tickers)


    def _get_estimated_sign(self, fc_df, em_prior_tickers):
        #Chicago + GS
        fc_df = pd.merge(pd.DataFrame([False] + fc_df.iloc[:-1].apply(lambda x: True if x < self._fc_threshold else False).tolist(),
                                            index = fc_df.index,
                                            columns = [self._fc_tickers[0]]),
                         pd.DataFrame(fc_df.apply(lambda x: True if x < self._fc_threshold else False).tolist(),
                                            index = fc_df.index,
                                            columns = [self._fc_tickers[1]]),
                         right_index=True, left_index=True)
        norm_fc_list = []
        for i in range(fc_df.shape[0]):
            if fc_df.index[i] < self._fc_label.index.min() or fc_df.index[i] > self._fc_label.index.max():
                norm_fc_list.append(fc_df[self._fc_tickers[0]].iloc[i])
            else:
                label_index =  int(self._fc_label.loc[fc_df.index[i]].iloc[0])
                if label_index == 2:
                    norm_fc_list.append(np.nan)
                else:
                    norm_fc_list.append(fc_df[self._fc_tickers[label_index]].iloc[i])

        #sign_df = pd.merge(em_prior_tickers, fc_prior_tickers, right_index=True, left_index=True)
        return pd.merge(em_prior_tickers, 
                        pd.DataFrame(norm_fc_list, index=fc_df.index, columns=['fc_priority']), 
                        right_index=True, left_index=True)


    def _get_indicated_sign(self, fc_df, em_prior_tickers):
        if self._has_indication_diff:#one week delay, like Chicago FC
            fc_prior_tickers = pd.DataFrame([False] + fc_df.iloc[:-1]\
                                                    .apply(lambda x: True if x < self._fc_threshold else False).tolist(),
                                            index = fc_df.index,
                                            columns = ['fc_priority'])
        else:
            fc_prior_tickers = pd.DataFrame(fc_df.apply(lambda x: True if x < self._fc_threshold else False).tolist(),
                                            index = fc_df.index,
                                            columns = ['fc_priority'])

        return pd.merge(em_prior_tickers, fc_prior_tickers, right_index=True, left_index=True)


    def simulate(self):
        self._logger.info("Simulation Starting...")
        rate_return_df = self._calc_return(self._price_df[self._em_rate_tickers].loc[self._date_list])
        fc_diff_df = self._price_df[self._fc_tickers].loc[self._date_list].diff().dropna(axis=0)
        src_return_df = pd.merge(rate_return_df, fc_diff_df, right_index=True, left_index=True)
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
    
        if self._use_estimated_sign:
            sign_df = self._get_estimated_sign(normalized_df[self._fc_tickers[0]], em_prior_tickers)
        else:
            if self._has_indication_diff:
                sign_df = self._get_indicated_sign(normalized_df[self._fc_tickers[0]], em_prior_tickers)
            else:
                sign_df = self._get_indicated_sign(fc_diff_df[self._fc_tickers[0]], em_prior_tickers)
            #import pdb;pdb.set_trace()
        self._logger.info("Building Position...")
        #Risk On: Long EM Ccy of Worst Score ->Position: -1(USD Short, EM Long)
        #of Worst
        #Risk OFF: Short EM Ccy of Best Score ->Position: 1(USD Long, EM Short)
        #of Best
        position_df = pd.DataFrame([(sign_df.iloc[i]['worst'], 0) 
                                    if np.isnan(sign_df.iloc[i]['fc_priority'])
                                    else (sign_df.iloc[i]['worst'], -1.0) 
                                        if sign_df.iloc[i]['fc_priority']
                                        else (sign_df.iloc[i]['best'], 1.0) 
                                     for i in range(sign_df.shape[0])],
                                     index = sign_df.index,
                                     columns=['ccy', 'ls'])
        position_df.index.name = 'ValueDate'
        if self._includes_swap:
            price_return_df = self._calc_return_inc_swap(self._price_df[self._price_tickers + [self._em_price_fwd_dic[k] 
                                                                        for k in self._em_price_fwd_dic.keys()]].loc[self._date_list],
                                                         self._price_tickers,
                                                         self._em_price_fwd_dic).loc[position_df.index]
        else:
            price_return_df = self._calc_return(self._price_df[self._price_tickers].loc[self._date_list], 
                                                with_log=True).loc[position_df.index]

        self._logger.info("Calculating Perofrmance...")
        return_series_df = pd.DataFrame([price_return_df[position_df.iloc[i][0]].iloc[i + 1] * position_df.iloc[i][1]
                                         for i in range(position_df.shape[0] - 1)],
                                        index = position_df.index[:-1],
                                        columns=['return'])
        return_series_df.index.name = 'ValueDate'
        return_series_df['cum_return'] = return_series_df['return'].cumsum()
        
        self._return_series_df = return_series_df
        self._sign_df = sign_df
        self._position_df = position_df
        self._fc_normalized_df = normalized_df[[self._fc_tickers[0]]]
        self._price_return_df = price_return_df

        self._logger.info("Simulation Complated.")


    def output(self, output_prefix='normal', output_suffix=None):
        self._logger.info("Outputting Result...")
        if output_suffix is None:
            output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')

        #Detailed Result
        self.output_detaild_result(self._position_df, self._return_series_df, output_prefix, output_suffix)

        #Return Series
        pd.merge(self._return_series_df, self._sign_df, right_index=True, left_index=True)\
          .to_csv(os.path.join('output', '{0}_em_reutrn_series_{1}.csv'.format(output_prefix,output_suffix)))

        #Performance Measure
        perform_measurer = PerformanceMeasurer()
        perform_measurer.create_result_summary(self._return_series_df)[['return']]\
            .to_csv(os.path.join('output','{0}_em_performance_{1}.csv'.format(output_prefix, output_suffix)))
        
        self._logger.info("Output Process Completed.")


    def output_detaild_result(self, position_df, return_series_df, output_prefix, output_suffix):
        merged_df = pd.merge(position_df, return_series_df, right_index=True, left_index=True)
        riskon_df = merged_df.query("ls < 0")
        riskoff_df = merged_df.query("ls > 0")
        riskon_hit_ratio = riskon_df["return"].apply(lambda x: 1 if x > 0 else 0).mean()
        riskoff_hit_ratio = riskoff_df["return"].apply(lambda x: 1 if x > 0 else 0).mean()

        date_df = pd.DataFrame(merged_df.index, columns = ['ValueDate'])
        
        riskon_df = pd.merge(date_df, riskon_df.reset_index('ValueDate'), 
                             on='ValueDate', how='left').set_index('ValueDate')
        riskoff_df = pd.merge(date_df, riskoff_df.reset_index('ValueDate'), 
                              on='ValueDate', how='left').set_index('ValueDate')

        perform_measurer = PerformanceMeasurer()
        riskon_performance = perform_measurer.create_result_summary(riskon_df[['return']].fillna(0.0))[['return']]
        riskoff_performance = perform_measurer.create_result_summary(riskoff_df[['return']].fillna(0.0))[['return']]
        riskon_performance.to_csv(os.path.join('output','{0}_em_riskon_{1}.csv'.format(output_prefix, output_suffix)))
        riskoff_performance.to_csv(os.path.join('output','{0}_em_riskoff_{1}.csv'.format(output_prefix, output_suffix)))

        riskon_df.fillna(method='ffill').to_csv(os.path.join('output', '{0}_em_riskon_series_{1}.csv'.format(output_prefix, output_suffix)))
        riskoff_df.fillna(method='ffill').to_csv(os.path.join('output', '{0}_em_riskoff_series_{1}.csv'.format(output_prefix, output_suffix)))
        
        pd.DataFrame([[riskon_hit_ratio, riskoff_hit_ratio],
                      [riskon_performance.T.MaxDD.iloc[0], riskoff_performance.T.MaxDD.iloc[0]],
                      [riskon_performance.T.AverageReturn.iloc[0], riskoff_performance.T.AverageReturn.iloc[0]],
                      [riskon_performance.T.Volatility.iloc[0], riskoff_performance.T.Volatility.iloc[0]]], 
                     index=['HitRatio', 'MaxDD', 'Return', 'Volatility'],
                     columns=['RiskOn', 'RiskOff']).to_csv(os.path.join('output', '{0}_detailed_result_{1}.csv'.format(output_prefix, output_suffix)))

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    roll_term = 104
    #start_date = date(2003, 3, 28)
    start_date = date(2006, 4, 7) - relativedelta(weeks=roll_term + 2)
    start_date = date(2007, 1, 1) - relativedelta(weeks=roll_term + 2)
    end_date = date.today()#date(2019, 12, 27)

    price_tickers = ['USDZAR Index', 'USDMXN Index']
    rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index']
    fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy']

    #price_tickers = ['USDZAR Index', 'USDMXN Index', 'USDTRY Index']
    #rate_tickers = ['GSAB2YR Index', 'GMXN02YR Index', 'GTRU2YR Index']
    #fwd_tickers = ['USDZAR1W BGN Curncy', 'USDMXN1W BGN Curncy', 'USDTRY1W BGN Curncy']
    import os
    #exp_return_file = os.path.join('input', 'em_ccy_expected_return_dnn_20203.csv')
    #fc_label = pd.read_csv(os.path.join('input', 'fc_label_hgb.csv'))
    #fc_label = cf.convert_date_format(fc_label)
    #fc_label.set_index('ValueDate', inplace=True)
    em_ccy_sim = EMCcySim(start_date=start_date, end_date=end_date, 
                          rolls=True,
                          roll_term=roll_term,
                          price_tickers=price_tickers,
                          em_rate_tickers=rate_tickers,
                          em_fwd_tickers=fwd_tickers,
                          #exp_return_file=exp_return_file,
                          use_estimated_sign=False,
                          #fc_label=fc_label
                          #fc_tickers=['GSUSFCI Index'],
                          #fc_tickers=['MXEF Index'],
                          #has_indication_diff=False
                          )
    em_ccy_sim.simulate()
    em_ccy_sim.output()