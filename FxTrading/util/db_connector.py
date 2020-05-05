#from RWAConfigParser import RWAConfigParser

import pymysql.cursors
import logging
import traceback
import pandas.io.sql as psql
from sqlalchemy import create_engine

import pandas as pd

class DBConnector(object):

    def __init__(self, *args, **kwargs):
        
        hostName = "localhost"
        userName = "root"
        loginPassword = "root"
        dbName = kwargs['DBName']

        self.__dbName = dbName
        con_url = "mysql+pymysql://" + userName + ":" + loginPassword + "@" + hostName + "/" + dbName
        self.__engine = create_engine(con_url, echo=False)
        self.__cnx = self.__engine.raw_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__cnx.close()
        self.__engine.dispose()

    
    def close(self):
        self.__cnx.close()
        self.__engine.dispose()

    def get_data(self, sqlString):
        data = pd.read_sql(sqlString, self.__cnx)

        return data

    def insert_data(self, df, tableName, exist_type='append'):
        fieldQuery = "SHOW FIELDS FROM " + tableName
        field_df = self.getData(fieldQuery)
        df.columns = field_df["Field"]
        #pd.io.sql.to_sql(name = tableName, con = self.__cnx, frame = df, if_exists = exist_type, flavor = 'mysql', index = False)
        df.to_sql(tableName, 
                         self.__engine, 
                         #if_exists = exist_type, 
                         #flavor='mysql',
                         if_exists=exist_type,
                         index = False)
                        

    def execute_query(self, sqlString):
        self.__engine.execute(sqlString)


if __name__ == "__main__":
    with DBConnector(DBName = "marketdb") as dbConn:
        sqlString = "SELECT * FROM equity_price"
        sqlResult = dbConn.getData(sqlString)
        print(str(sqlResult))
        #dbConn.closeConnection()

