# -*- coding: utf-8 -*-
import os, sys
import logging.config
import gc

import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import shutil

from util.ml_config_parser import MLConfigParser
import algorithm as alg

from util.feature_vector_manager import FeatureVectorManager
from util.result_manager import ResultManager
import util.common_func as cf

def check_input(target_file):
    if not os.path.exists(target_file): 
        raise Exception('{0} does not exist.'.format(target_file))


def create_date_list(input_file_name, term_month):

    input_data_df = pd.read_csv(input_file_name)
    input_data_df['ValueDate'] = pd.to_datetime(input_data_df.ValueDate.values)#cf.convert_date_format(input_data_df.ValueDate)
    input_data_df.set_index('ValueDate', inplace=True)
    start_date = np.min(list(input_data_df.index)) + relativedelta(months=term_month)
    return input_data_df.query("ValueDate >= @start_date").index


def calc_training_result(algo, training_data, training_label, value_date):
    training_result = algo.predict(training_data)
    
    training_result_df = pd.DataFrame(training_result, columns=['Predict'])
    if training_result_df.shape[0] == training_data.shape[0]:
        training_result_df['ValueDate'] = training_data.index
    else:
        training_result_df['ValueDate'] = training_data.index[-training_result_df.shape[0]:]
    training_result_df['Algorithm'] = algo.__class__.__name__
    training_result_df.set_index('ValueDate', inplace=True)
    
    result_manager = ResultManager(PredictedData=training_result_df,
                                   PredictedLabel=training_label.loc[training_result_df.index],
                                   multi_class=False)
    summary_df = result_manager.create_result(type='Training')
    summary_df['ValueDate'] = value_date
    
    return summary_df


def create_label(input_file_name):
    '''return label dataframe (Return column) for back test.
       Parameters
       ----------
       input_file_name: csv file of weekly data which has weekly date column
    '''
    input_df = pd.DataFrame(pd.read_csv(input_file_name))
    input_df['ValueDate'] = pd.to_datetime(input_df.ValueDate.values)#cf.convert_date_format(input_df.ValueDate)
    return pd.DataFrame(input_df.set_index('ValueDate').Return)


def main():
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    
    try:
        config = MLConfigParser()
        training_month = config.training_term
        exec_pca = config.exec_pca
        is_regression = config.is_regression
        
        feature_file_name = os.path.join(config.input_dir, config.feature_file)
        output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')     

        check_input(feature_file_name)
        logger.info("Trainig Term {0}".format(training_month))
        logger.info("Excecute PCA {0}".format(exec_pca))
        date_list = create_date_list(feature_file_name, 
                                      training_month)
    
        algorithm_list = [
                          alg.ML_DNN,
                          #alg.ML_CNN,
                          #alg.ML_RNN,
                          #alg.ML_LSTM,
                          #alg.ML_GRU,
                          ]
        
        predict_result_df = pd.DataFrame()
        proba_result_df = pd.DataFrame()
        training_result_df = pd.DataFrame()
        importance_df = pd.DataFrame()

        for algo in algorithm_list:
            algo_result_list = []
            proba_result_list = []
            ml_algo = algo(IsRegression=is_regression, with_grid_cv=config.with_grid_cv)
            prev_date = None
            
            for value_date in tqdm(date_list):
                logger.info("Trainig/Predicting In {0}...".format(value_date))

                #fix start date and expand training data, or roll training data
                if config.fix_start_date:
                    start_date = date_list[0] - relativedelta(months=training_month)
                else:
                    start_date = value_date - relativedelta(months=training_month)
            
                feature_manager = FeatureVectorManager(FilePath=feature_file_name, 
                                                        TrainingStartDate=start_date-relativedelta(weeks=1),
                                                        PredictStartDate=value_date,
                                                        PredictEndDate=value_date,
                                                        IsRegression=is_regression,
                                                        ExecPCA=exec_pca,
                                                        scaler_type=config.scaler_type,
                                                        select_feature=False,
                                                        multi_class=True,
                                                        MaxLen=None if 'ml_time_series' not in ml_algo.__module__  and 'ml_cnn' not in ml_algo.__module__ else ml_algo.maxlen)
                #if prev_date is None or prev_date.year != value_date.year:
                logger.info("Learing In {0}".format(value_date))
                ml_algo.dispose()
                ml_algo.learn(training_data=feature_manager.training_data,
                                training_label=feature_manager.training_label, # label_df.reindex(feature_manager.training_data.index),
                                tunes_param=config.parameter_tuning)
                prev_date = value_date
                #if 'ml_time_series' not in ml_algo.__module__:
                training_result_df = training_result_df.append(
                                        calc_training_result(ml_algo,feature_manager.training_data,
                                                              feature_manager.training_label,
                                                              value_date))
                algo_result_list.append([value_date,
                                          ml_algo.__class__.__name__,
                                          ml_algo.predict_one(feature_manager.predict_data)])
                
                #proba_result_list.append([value_date, ml_algo.__class__.__name__]
                #                          +ml_algo.predict_one_proba(feature_manager.predict_data))
                
                #Post Process for each week
                feature_manager.dispose()
                del feature_manager
                gc.collect()

            predict_result_df = predict_result_df.append(
                                    pd.DataFrame(algo_result_list,
                                                  index=date_list, 
                                                  columns=['ValueDate',
                                                          'Algorithm',
                                                          'Predict']))
            #if not is_regression:
            #    proba_result_df = proba_result_df.append(
            #                            pd.DataFrame(proba_result_list,
            #                                          index=date_list, 
            #                                          columns=['ValueDate',
            #                                                  'Algorithm',
            #                                                  'DownProbability',
            #                                                  'UpProbability']))
        #Result Output Process
        predict_result_df.index.name='ValueDate'
        #proba_result_df.index.name='ValueDate'
        predict_result_file = 'predict_result_{0}_{1}_{2}_{3}.csv'.format('PCA' if exec_pca else 'NoPCA',
                                                                          int(training_month),
                                                                          'Reg' if is_regression else 'Class',
                                                                          output_suffix)
        
        predict_result_df.to_csv(os.path.join(config.output_dir, 
                                              predict_result_file), 
                                 index=False)
        shutil.copy2(os.path.join(config.output_dir, predict_result_file),
                     os.path.join(config.input_dir, config.fc_label_file))

        if not is_regression:
            proba_result_df.to_csv(os.path.join(config.output_dir, 'proba_result_{0}_{1}_{2}_{3}.csv')
                                    .format('PCA' if exec_pca else 'NoPCA',
                                            int(training_month),
                                            'Reg' if is_regression else 'Class',
                                            output_suffix), 
                                        index=False)
        training_result_df.to_csv(os.path.join(config.output_dir, 'training_result_{0}_{1}_{2}_{3}.csv')
                                    .format('PCA' if exec_pca else 'NoPCA',
                                            int(training_month),
                                            'Reg' if is_regression else 'Class',
                                            output_suffix), 
                                        index=False)

        result_manager = ResultManager(PredictedData=predict_result_df,
                                       PredictedLabel=create_label(feature_file_name),
                                       multi_class=False)
        result_manager.create_result().to_csv('./output/summary_{0}_{1}_{2}_{3}.csv'
                                                .format('PCA' if exec_pca else 'NoPCA',
                                                        int(training_month),
                                                        'Reg' if is_regression else 'Class',
                                                        output_suffix), 
                                                index=False)
    except:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
