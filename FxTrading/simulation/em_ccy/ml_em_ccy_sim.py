# -*- coding: utf-8 -*-
import os, sys
import logging.config
import gc

import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from util.ml_config_parser import MLConfigParser
from exception.exceptions import InvalidFileError
import algorithm as alg

from util.feature_vector_manager import FeatureVectorManager
from util.result_manager import ResultManager
import util.common_func as cf

def check_input(target_file):
    if not os.path.exists(target_file): 
        raise InvalidFileError('{0} does not exist.'.format(target_file))


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
                                   PredictedLabel=training_label.loc[training_result_df.index])
    #import pdb;pdb.set_trace()
    summary_df = result_manager.create_result(type='Training')
    summary_df['ValueDate'] = value_date
    
    return summary_df

def create_importance(algo_name, importance, columns, value_date):
    importance_df = pd.DataFrame(importance).T
    importance_df.columns = columns
    importance_df['Algorithm'] = algo_name
    importance_df.index = [value_date]

    return importance_df

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
        #feature_file_list = ['factor_ZAR.csv']#, 'factor_MXN.csv', 'factor_TRY.csv']
        #feature_file_list = ['factor_ZAR.csv']
        feature_file_list = [config.feature_file]
        for feature_file_name in feature_file_list:
            feature_file_name = os.path.join('input', feature_file_name)
            output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')     

            check_input(feature_file_name)
            logger.info("Trainig Term {0}".format(training_month))
            logger.info("Excecute PCA {0}".format(exec_pca))
            date_list = create_date_list(feature_file_name, 
                                         training_month)
        
            algorithm_list = [
                              #alg.ML_Adaboost, 
                              #alg.ML_Bagging, 
                              #alg.ML_GradientBoost, 
                              #alg.ML_SVM,
                              #alg.ML_RandomForest,
                              #alg.ML_HistGradientBoost,
                              #alg.ML_XGBoost,
                              #alg.ML_LightGBM,
                              #alg.ML_kNN,
                              #alg.ML_LinearRegression,
                              #alg.ML_RidgeRegression,
                              #alg.ML_LassoRegression,
                              #alg.ML_ElasticNet,
                              #alg.ML_BasianRegression,
                              #alg.ML_ARDRegression,
                              #alg.ML_DNN,
                              #alg.ML_DNN_TF,
                              #alg.ML_CNN,
                              #alg.ML_CNN_TF,
                              #alg.ML_LSTM_TF,
                              alg.ML_RNN_TF,
                              #alg.ML_RNN,
                              alg.ML_GRU_TF,
                              #alg.ML_GRU,
                              ]
            #if config.is_regression:
            #    algorithm_list.append(alg.ML_NaiveBayes)
            #algorithm_list = [alg.ML_LightBGM]

            predict_result_df = pd.DataFrame()
            proba_result_df = pd.DataFrame()
            training_result_df = pd.DataFrame()
            importance_df = pd.DataFrame()
        
            for algo in algorithm_list:
                algo_result_list = []
                proba_result_list = []
                #date_list = date_list[date_list>date(2018,12,1)]
                #date_list = date_list[-100:]
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
                                                           TrainingStartDate=start_date,
                                                           PredictStartDate=value_date,
                                                           PredictEndDate=value_date,
                                                           IsRegression=is_regression,
                                                           ExecPCA=exec_pca,
                                                           scaler_type=config.scaler_type,
                                                           select_feature=False,
                                                           MaxLen=None if 'ml_time_series' not in ml_algo.__module__  and 'ml_cnn' not in ml_algo.__module__ else ml_algo.maxlen)
                    #if prev_date is None or prev_date.year != value_date.year:
                    logger.info("Learing In {0}".format(value_date))
                    ml_algo.dispose()
                    ml_algo.learn(training_data=feature_manager.training_data,
                                    training_label=feature_manager.training_label,
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
                    
                    #if not is_regression and ml_algo.__class__.__module__[-10:] != 'regression':
                    #    proba_result_list.append([value_date, ml_algo.__class__.__name__]\
                    #                             +ml_algo.predict_one_proba(feature_manager.predict_data))

                    if ml_algo.__class__.__name__ in config.importance_models:
                        importance_df = importance_df.append(create_importance(ml_algo.__class__.__name__,
                                                                               ml_algo.importance,
                                                                               feature_manager.training_data.columns,
                                                                               value_date))

                    #Post Process for each week
                    feature_manager.dispose()
                    #ml_algo.dispose()
                    #del ml_algo
                    del feature_manager
                    gc.collect()

                predict_result_df = predict_result_df.append(
                                        pd.DataFrame(algo_result_list,
                                                     index=date_list, 
                                                     columns=['ValueDate',
                                                              'Algorithm',
                                                              'Predict']))
                if not is_regression:
                    proba_result_df = proba_result_df.append(
                                            pd.DataFrame(proba_result_list,
                                                         index=date_list, 
                                                         columns=['ValueDate',
                                                                  'Algorithm',
                                                                  'DownProbability',
                                                                  'UpProbability']))
            #Result Output Process
            predict_result_df.index.name='ValueDate'
            proba_result_df.index.name='ValueDate'
            predict_result_df.to_csv('./output/predict_result_{0}_{1}_{2}_{3}.csv'\
                                    .format('PCA' if exec_pca else 'NoPCA',
                                            int(training_month),
                                            'Reg' if is_regression else 'Class',
                                            output_suffix), 
                                     index=False)
            if not is_regression:
                proba_result_df.to_csv('./output/proba_result_{0}_{1}_{2}_{3}.csv'\
                                        .format('PCA' if exec_pca else 'NoPCA',
                                                int(training_month),
                                                'Reg' if is_regression else 'Class',
                                                output_suffix), 
                                         index=False)
            training_result_df.to_csv('./output/training_result_{0}_{1}_{2}_{3}.csv'\
                                    .format('PCA' if exec_pca else 'NoPCA',
                                            int(training_month),
                                            'Reg' if is_regression else 'Class',
                                            output_suffix), 
                                     index=False)
    
            importance_df.to_csv('./output/importance_{0}_{1}_{2}_{3}.csv'\
                                 .format('PCA' if exec_pca else 'No',
                                         int(training_month),
                                         'Reg' if is_regression else 'Class',
                                         output_suffix), 
                                 index=True)
            result_manager = ResultManager(PredictedData=predict_result_df,
                                           PredictedLabel=create_label(feature_file_name))
            result_manager.create_result().to_csv('./output/summary_{0}_{1}_{2}_{3}.csv'\
                                                   .format('PCA' if exec_pca else 'NoPCA',
                                                           int(training_month),
                                                           'Reg' if is_regression else 'Class',
                                                           output_suffix), 
                                                  index=False)
            sys.exit(0)
    except InvalidFileError as ife:
        logger.error(ife.args)
    else:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
