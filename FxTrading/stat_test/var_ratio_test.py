# -*- coding: utf-8 -*-
import logging
import gc

import pandas as pd
import numpy as np

from stat_test.base_test  import BaseTest

class VarRatioTest(BaseTest):
    def __init__(self, *args, **kwargs):
        #import pdb;pdb.set_trace()
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._term = kwargs.get('term', 261)
        self._num = kwargs.get('num', 2)

        print(self._return_df.shape[0])
        assert self._return_df.shape[0] >= (self._term * self._num)

        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def exec(self):
        
        result_df = pd.DataFrame()
        for i in range(self._term*self._num, self._return_df.shape[0]):
            self._logger.info("Processing in {0}...".format(self._return_df.index[i]))
            target_df = self._return_df.iloc[i-self._term*self._num:i]
            avg_df = pd.DataFrame(target_df.mean(axis=0)).T
            var_s = pd.DataFrame(((np.array(target_df) - np.array(avg_df))**2).sum(axis=0) \
                                / (self._term*self._num-1)).T
            param_w = self._term * (self._num * self._term - self._term + 1)*(1 - 1/self._num)

            var_l = pd.DataFrame()
            
            for j in range(i-self._term*self._num+self._term, i):
                
                var_l = var_l.append(pd.DataFrame(np.array((target_df.iloc[j-self._term:j].sum(axis=0) \
                                - self._term * avg_df)**2)))
            
            var_l = var_l.sum(axis=0) / param_w
            
            #import pdb; pdb.set_trace()
            result_df = result_df.append(np.sqrt(self._num*self._term)*(var_l/var_s - 1) \
                                    * (2*(2*self._term-1)*(self._term-1)/(3*self._term))**(-0.5))

        result_df.columns = self._return_df.columns
        result_df.index = self._return_df.index[self._term*self._num:]
        return result_df


if __name__ == '__main__':
    import logging.config
    from datetime import date
    logging.config.fileConfig('./logger_config.ini')
    start_date = date(2009, 1, 1)
    end_date = date(2019, 5, 24)
    
    run_test = VarRatioTest(start_date=start_date, end_date=end_date)
    run_test.exec().to_csv('var_ratio_test.csv')
    