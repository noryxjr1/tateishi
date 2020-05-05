# -*- coding: utf-8 -*-
import logging
import gc

import pandas as pd
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from stat_test.base_test  import BaseTest

class RunTest(BaseTest):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._term = kwargs.get('term', 30)
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    def exec_r(self):
        import pyper
        r = pyper.R(use_pandas='True')
        r("suppressWarnings(require(tseries,warn.conflicts = FALSE,quietly=TRUE))")
        #r("source(file='mswm.R')")#, encoding='utf-8'
        #r("result<-m.lm$coefficients[2]")
        #r("result1<-y1")
        #r("result2<-y2")
        #print(r.get("result"))
        #print("completed")

        for i in range(self._term, self._return_df.shape[0]):
            self._return_df.iloc[i-self._term:i].to_csv('run.csv')
            r("df <- read.csv('run.csv', header = T)")
            r("d <- diff(df$Last)")
            r("x <- factor(sign(d[-which(d %in% 0)]))")
            r("run_result <- runs.test(x)")
            r("p_value <- run_rsult$p.value")
            print(r.get('p_value'))

    def exec(self):
        
        result_df = pd.DataFrame()
        for i in range(self._term, self._return_df.shape[0]):
            target_df = self._return_df.iloc[i-self._term:i]
            self._logger.info("Processing in {0}...".format(target_df.index[-1]))
            relative_return_df = target_df - target_df.mean(axis=0)
            
            plus_df = pd.DataFrame([np.where(relative_return_df > 0, 1, 0).sum(axis=0)],
                                   columns=target_df.columns)
            minus_df = pd.DataFrame([np.where(relative_return_df < 0, 1, 0).sum(axis=0)],
                                    columns=target_df.columns)

            exp_return_df = 2*plus_df*minus_df / (plus_df+minus_df) + 1
            exp_vol_df = pd.DataFrame(np.sqrt(2*plus_df*minus_df*(2*plus_df*minus_df - plus_df - minus_df) \
                                    / ((plus_df + minus_df)**2 * (plus_df+minus_df-1))),
                                      columns=target_df.columns)
        
            run_df = self._calc_run(relative_return_df)
            result_df = result_df.append((run_df - np.array(exp_return_df) + 0.5) / np.array(exp_vol_df))

        return result_df


    def _calc_run(self, relative_return_df):
        return pd.DataFrame([np.where(np.array(relative_return_df.iloc[1:]) \
                                   * np.array(relative_return_df.iloc[:-1])<0,1,0).sum(axis=0)],
                            index = [relative_return_df.index[-1]],
                            columns = relative_return_df.columns)

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('./logger_config.ini')
    start_date = date(2004, 1, 1)
    end_date = date(2019, 5, 24)
    term = 750
    run_test = RunTest(start_date=start_date, end_date=end_date)
    run_test.exec_r()
    #run_test.exec().to_csv('run_test_{0}days.csv'.format(term))
    