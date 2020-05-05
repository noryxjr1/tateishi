"""This Commdule has common function for the project"""
# -*- coding: utf-8 -*-
from datetime import date
from functools import lru_cache

import util

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
    config = util.MLConfigParser()
    date_query = "SELECT ValueDate FROM bbg_marketprice \
                  WHERE Ticker = 'USDJPY Index' \
                  AND ValueDate BETWEEN '{0}' AND '{1}' AND last IS NOT NULL "\
                  .format(start_date,
                          end_date)
    with util.DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(date_query).ValueDate.tolist()


@lru_cache()
def create_weekly_datelist(start_date=date(2001, 4, 1),
                           end_date=date.today(),
                           weeknum=6):
    config = util.MLConfigParser()
    date_query = "SELECT ValueDate FROM bbg_marketprice \
                          WHERE DAYOFWEEK(ValueDate) = {2} \
                          AND ValueDate BETWEEN '{0}' AND '{1}' \
                          AND Ticker = 'USDJPY Index'"\
                          .format(start_date, 
                                  end_date,
                                  weeknum)
    with util.DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(date_query).ValueDate.tolist()


@lru_cache()
def create_monthly_datelist(start_date=date(2001, 4, 1),
                            end_date=date.today()):
    config = util.MLConfigParser()
    date_query = "SELECT MAX(ValueDate) AS ValueDate \
                  FROM bbg_marketprice \
                  WHERE Ticker = 'USDJPY Index' \
                  AND ValueDate BETWEEN '{0}' AND '{1}' \
                  AND last IS NOT NULL\
                  GROUP BY Ticker, YEAR(ValueDate), MONTH(ValueDate)"\
                  .format(start_date,
                          end_date)
    with util.DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(date_query).ValueDate.tolist()


def get_fx_rate(start_date=date(2001,1,1), 
                end_date=date.today(), 
                ccy_list=['USDJPY Index','EURJPY Index']):
    fx_query = "SELECT ValueDate, Ticker, Last \
                FROM bbg_marketprice \
                WHERE ValueDate BETWEEN '{0}' AND '{1}' \
                AND Ticker IN ('{2}')".format(start_date, end_date, "','".join(ccy_list))

    with util.DBConnector(DBName='marketdb') as db_conn:
        return db_conn.get_data(fx_query).pivot(index='ValueDate', 
                                                columns='Ticker', 
                                                values='Last')


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

