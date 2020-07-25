import os
from fc_selector.db.db_connector import DBConnector


if __name__ == '__main__':
    with DBConnector(DBName='marketdb') as db_conn:
        db_conn.get_data("""SELECT * FROM marketdb.bbg_marketprice 
                            where Ticker IN ('USDJPY Index',
                                             'USDZAR Index',
                                             'USDMXN Index',
                                             'USDZAR1W BGN Curncy', 
                                             'USDMXN1W BGN Curncy',
                                             'NFCIINDX Index',
                                             'GSUSFCI Index',
                                             'USGG2YR Index',
                                             'GSAB2YR Index',
                                             'GMXN02YR Index',
                                             'GTRU2YR Index',
                                             'CESIUSD Index',
                                             'CECIUSD Index',
                                             'CTOTUSD Index',
                                             'CESIZAR Index',
                                             'CECIZAR Index',
                                             'CTOTZAR Index',
                                             'CESIMXN Index',
                                             'CECIMXN Index',
                                             'CTOTMXN Index',
                                             'BISBUSN Index',
                                             'BISBZAN Index',
                                             'BISBMXN Index')""").to_csv(os.path.join('input', 'all_input_data.csv'), index=False)
