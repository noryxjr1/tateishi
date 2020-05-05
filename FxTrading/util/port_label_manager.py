import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from util.db_connector import DBConnector
import util.common_func as cf

class PortLabelManager(object):
    def __init__(self, **kwargs):
        term = kwargs.get('term', 1)
        weight_df = pd.read_csv(kwargs.get('weight_file', 
                                           './input/coint_vec.csv'))
        price_df = pd.read_csv(kwargs.get('price_file', 
                                          './input/index_price_52.csv'))
        price_df = cf.convert_date_format(price_df).set_index('ValueDate').iloc[:, -1]
        
        weight_df['ValueDate'] = cf.convert_date_format(weight_df)
        weight_df.set_index('ValueDate', inplace=True)

        src_df = np.log(cf.get_fx_rate(start_date = weight_df.index[0],
                                       end_date = weight_df.index[-1],
                                       ccy_list = weight_df.columns).loc[weight_df.index])
        
        assert src_df.shape[0] == weight_df.shape[0]
        
        self._price_dic = self._create_price_index(weight_df, src_df)
        #import pdb;pdb.set_trace()
        port_label = self._create_port_label(weight_df, src_df, term).loc[price_df.index]
        self._notional = pd.DataFrame((np.abs(weight_df) * src_df).loc[price_df.index].sum(axis=1), columns=['Notional'])
        self._port_label = pd.DataFrame(port_label.Return / self._notional.Notional,
                                        columns=['Return'])


    @property
    def price_series(self):
        return self._price_dic

    @property
    def port_label(self):
        return self._port_label

    @property
    def notional(self):
        return self._notional

    #def get_target_cointvec(self,
    #                         pvalue_file='./input/pvalue_matrix.csv',
    #                         cointvec_file='./input/coint_weight.csv'):
    #    ticker_list = ['USDJPY Index', 'EURJPY Index', 'AUDJPY Index', 'GBPJPY
    #    Index',
    #                   'CADJPY Index', 'CHFJPY Index', 'NZDJPY Index']
    #    pvalue_df =
    #    cf.convert_date_format(pd.read_csv(pvalue_file)).set_index('ValueDate')
    #    weight_df =
    #    cf.convert_date_format(pd.read_csv(cointvec_file)).set_index('ValueDate')
        
    #    result_df = pd.DataFrame()
    #    for value_date in pvalue_df.index:
    #        target_ccy = pvalue_df.query("index == @value_date").T.idxmin()[0]
    #        coint_df = pd.DataFrame(weight_df.query("index ==
    #        @value_date")).query("Portfolio == @target_ccy")
            
    #        one_df=pd.DataFrame([np.ones(len(ticker_list))],
    #                            index=['Weight'],
    #                            columns=ticker_list)
            
    #        result_df = result_df.append((one_df *
    #        coint_df.reset_index('ValueDate')\
    #                                                       .set_index('Ccy')[['Weight']].T)\
    #                                                       .fillna(0.0))
        
    #    result_df.index = pvalue_df.index
    #    return result_df


    def _create_port_label(self, weight_df, src_df, term):
        #import pdb;pdb.set_trace()
        #label_list = []
        #for i in range(weight_df.shape[0]-term):
        #    label_list.append(((np.array(src_df.iloc[i+term]) \
        #                     - np.array(src_df.iloc[i]))\
        #                     *np.array(weight_df.iloc[i])).sum())


        return pd.DataFrame([((np.array(src_df.iloc[i + term]) - np.array(src_df.iloc[i])) 
                              * np.array(weight_df.iloc[i])).sum() 
                             for i in range(weight_df.shape[0] - term)], 
                            index=weight_df.index[:-term], 
                            columns=['Return'])


    def _create_price_index(self, weight_df, src_df):
        price_dic = {}
        for i in range(weight_df.shape[0]):
            price_dic[weight_df.index[i]] = pd.DataFrame((src_df * np.array(weight_df[src_df.columns].iloc[i])).sum(axis=1),
                                                         columns=['Price'])

        return price_dic
        #return pd.DataFrame((src_df * weight_df[src_df.columns]).sum(axis=1),
        #                    columns=['Price'])


if __name__ == '__main__':
    port_label_mgr = PortLabelManager()
    #port_label_mgr.get_target_cointvec(pvalue_file='pvalue_matrix_20191014191718.csv',
    #                                   cointvec_file='coint_weight_result_20191014191718.csv').to_csv('coint_vec.csv')
    print(port_label_mgr.port_label)




