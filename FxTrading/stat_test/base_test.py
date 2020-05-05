# -*- coding: utf-8 -*-
import logging
import gc

import pandas as pd
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from abc import ABCMeta, abstractmethod

import Util
class BaseTest(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("nory.xjr1")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._end_date = kwargs.get('end_date', date.today())
        self._start_date = kwargs.get('start_date', self._end_date - relativedelta(years=1))

        self._ccy_list = kwargs.get('ccy_list', ['EURUSD Index',
                                                'USDJPY Index',
                                                'EURJPY Index',
                                                'GBPJPY Index',
                                                'AUDJPY Index',
                                                'NZDJPY Index',
                                                'CHFJPY Index',])
        self._fx_rate = self._get_fx_rate()
        self._return_df = pd.DataFrame(np.log(np.array(self._fx_rate.iloc[1:])) \
                                     - np.log(np.array(self._fx_rate.iloc[:-1])),
                                       index = self._fx_rate.index[:-1],
                                       columns = self._fx_rate.columns)
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    def _get_fx_rate(self):
        rate_query = "SELECT ValueDate, Ticker, Last \
                    FROM bbg_marketprice \
                   WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                     AND Ticker IN ('{2}') "\
                 .format(self._start_date, self._end_date, "','".join(self._ccy_list))
        with Util.DBConnector(DBName='marketdb') as db_conn:
            return db_conn.getData(rate_query).pivot(index='ValueDate',
                                                     columns='Ticker',
                                                     values='Last')\
                                              .fillna(method='ffill')

    @abstractmethod
    def exec(self, term=30):
        pass
    