"""This Commdule has common function for the project"""
# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import date
from functools import lru_cache



def convert_date_format(src_vector):
    if '-' in src_vector.iloc[0]:
        return src_vector.apply(lambda x: date(int(x.split('-')[0]),
                                               int(x.split('-')[1]),
                                               int(x.split('-')[2])))
    elif '/' in src_vector.iloc[0]:
        return src_vector.apply(lambda x: date(int(x.split('/')[0]),
                                               int(x.split('/')[1]),
                                               int(x.split('/')[2])))


@lru_cache()
def create_daily_datelist(start_date=date(2001, 4, 1),
                          end_date=date.today()):
    input_df = convert_date_format(pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                            '../input', 'all_input_data.csv')))
    return input_df.query("Ticker == 'USDJPY Index' & @start_date <= ValueDate <= @end_date").ValueDate.tolist()


@lru_cache()
def create_weekly_datelist(start_date=date(2001, 4, 1),
                           end_date=date.today(),
                           weeknum=6):
    input_df = convert_date_format(pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                            '../input', 'all_input_data.csv')))
    daily_date = input_df.query("Ticker == 'USDJPY Index' & @start_date <= ValueDate <= @end_date").ValueDate
    
    return daily_date[daily_date.apply(lambda x: True if x.weekday()==4 else False)].tolist()


@lru_cache()
def create_monthly_datelist(start_date=date(2001, 4, 1),
                           end_date=date.today()):
    input_df = convert_date_format(pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                            '../input', 'all_input_data.csv')))
    daily_date = input_df.query("Ticker == 'USDJPY Index' & @start_date <= ValueDate <= @end_date").ValueDate
    
    return daily_date[[True if daily_date.iloc[i].month != daily_date.iloc[i+1].month else False 
                       for i in range(daily_date.shape[0]-1)] + [False]].tolist()


def convert_date_format(input_vector, target_col='ValueDate'):
        if '/' in input_vector.ValueDate.iloc[0]:
            input_vector[target_col] = input_vector[target_col].apply(lambda x: date(int(x.split('/')[0]), 
                                                                                    int(x.split('/')[1]), 
                                                                                    int(x.split('/')[2])))
        else:
            input_vector[target_col] = input_vector[target_col].apply(lambda x: date(int(x.split('-')[0]), 
                                                                                    int(x.split('-')[1]), 
                                                                                    int(x.split('-')[2])))
        return input_vector

if __name__ == '__main__':
    create_monthly_datelist()