# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:34:00 2018
@author: jpbank.quants
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from algorithm.ml_base import ML_Base
import util.common_func as cf

class ML_AutoRegBase(ML_Base):
    cv_model = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._start_date = kwargs.get('start_date', None)
        self._end_date = kwargs.get('end_date', None)
        interval = kwargs.get('interval', 1)
        frequency = kwargs.get('frequency', 'weekly')
        if frequency == 'daily':
            date_list = cf.create_daily_datelist(self._start_date, self._end_date)
        elif frequency == 'weekly':
            date_list = cf.create_weekly_datelist(self._start_date, self._end_date)
        else:
            date_list = cf.create_monthly_datelist(self._start_date, self._end_date)

        coint_vec_file = kwargs.get('coint_Vec_file', './input/coint_vec.csv')
        self._weight_df = cf.convert_date_format(pd.read_csv(coint_vec_file))\
                            .set_index('ValueDate').loc[date_list]

        self._fx_rate_df = np.log(cf.get_fx_rate(self._start_date,
                                                 self._end_date,
                                                 self._weight_df.columns.tolist())).loc[date_list]
        #import pdb;pdb.set_trace()
        self._coint_index_df = pd.DataFrame((self._fx_rate_df*self._weight_df[self._fx_rate_df.columns]).sum(axis=1),
                                            columns=['Price']).loc[date_list]

        #self._index_return_df = pd.DataFrame(self._coint_index_df[interval:] - self._coint_index_df[:-interval],
        #                                    index = self._fx_rate_df.index[:-interval],
        #                                    columns = ['Return'])#.loc[date_list]
        
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def coint_index(self):
        return self._coint_index_df

    @property
    def param(self):
        return self._estimated_result


    def learn(self, start_date, end_date):
        pass
        ##sm.tsa.arma_order_select_ic(training_label, ic='aic', trend='nc')
        #target_fx_rate = self._fx_rate_df.query("index>=@start_date & index<=@end_date")
        #target_weight = self._weight_df.query("index<=@end_date").iloc[0]
        #coint_index_df = pd.DataFrame((target_fx_rate * target_weight).sum(axis=1),
        #                              columns=['Price'])
        ##self._model = sm.tsa.AR(coint_index_df)
        #self._model = sm.tsa.ARMA(endog=coint_index_df, order=(3, 6))
        # #ARMA, ARIMA
        ##self._model.select_order(maxlag=3, ic='aic')
        #self._estimated_result = self._model.fit(maxlag=3, trend='c', maxiter=1000, solver='powell')
        ##import pdb;pdb.set_trace()
        
    def predict(self, value_date):
        #import pdb;pdb.set_trace()
        return self._model.predict(self._estimated_result.params, value_date)[-1]



class ML_AR(ML_AutoRegBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, start_date, end_date):
        target_fx_rate = self._fx_rate_df.query("index>@start_date & index<=@end_date")
        target_weight = self._weight_df.query("index==@end_date").iloc[0]
        coint_index_df = pd.DataFrame((target_fx_rate * target_weight).sum(axis=1),
                                      columns=['Price'])
        self._model = sm.tsa.AR(coint_index_df)
        self._model.select_order(maxlag=3, ic='aic')
        self._estimated_result = self._model.fit(maxlag=3)



class ML_ARMA(ML_AutoRegBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, start_date, end_date):
        target_fx_rate = self._fx_rate_df.query("index>=@start_date & index<=@end_date")
        target_weight = self._weight_df.query("index<=@end_date").iloc[0]
        coint_index_df = pd.DataFrame((target_fx_rate * target_weight).sum(axis=1),
                                      columns=['Price'])
        self._model = sm.tsa.ARMA(endog=coint_index_df, order=(3, 6))
        self._estimated_result = self._model.fit(maxlag=3, trend='c', maxiter=1000, solver='powell')


class ML_ARIMA(ML_AutoRegBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, start_date, end_date):
        target_fx_rate = self._fx_rate_df.query("index>=@start_date & index<=@end_date")
        target_weight = self._weight_df.query("index<=@end_date").iloc[0]
        coint_index_df = pd.DataFrame((target_fx_rate * target_weight).sum(axis=1),
                                      columns=['Price'])
        self._model = sm.tsa.ARIMA(coint_index_df, order=(3, 2, 6))
        self._estimated_result = self._model.fit(maxlag=3, trend='nc', maxiter=1000, solver='powell')
