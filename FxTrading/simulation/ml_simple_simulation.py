# -*- coding: utf-8 -*-
import os, sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import logging.config
import gc

import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pickle as pkl

from util.ml_config_parser import MLConfigParser
from exception.exceptions import InvalidFileError
import algorithm as alg

from util.feature_vector_manager import FeatureVectorManager
from util.result_manager import ResultManager

def check_input(target_file):
    if not os.path.exists(target_file): 
        raise InvalidFileError('{0} does not exist.'.format(target_file))


def calc_training_result(algo, training_data, training_label, value_date):
    training_result = algo.predict(training_data)
    training_result_df = pd.DataFrame(training_result, columns=['Predict'])
    #import pdb;pdb.set_trace()

    if training_result_df.shape[0] == training_data.shape[0]:
        training_result_df['ValueDate'] = training_data.index
    else:
        training_result_df['ValueDate'] = training_data.index[-training_result_df.shape[0]:]

    training_result_df['Algorithm'] = algo.__class__.__name__
    result_manager = ResultManager(PredictedData=training_result_df.set_index('ValueDate'),
                                   PredictedLabel=training_label)
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
    input_df['ValueDate'] = input_df\
                            .ValueDate.apply(lambda x: date(int(x.split('/')[0]),
                                                            int(x.split('/')[1]),
                                                            int(x.split('/')[2])))
    #input_df['ValueDate'] = pd.to_datetime(input_df.ValueDate.values)
    return pd.DataFrame(input_df.set_index('ValueDate').Return)


def main():
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
    try:
        config = MLConfigParser()
        training_month = config.training_term
        exec_pca = config.exec_pca
        is_regression = config.is_regression
        input_file_name = os.path.join(config.input_dir, config.feature_file)
        #feature_file_name = os.path.join(config.input_dir, config.feature_file)

        check_input(input_file_name)
        #check_input(feature_file_name)

        logger.info("Trainig Term {0}".format(training_month))
        logger.info("Excecute PCA {0}".format(exec_pca))
        
        algorithm_list = [
                          #alg.ML_Adaboost, 
                          #alg.ML_Bagging, 
                          #alg.ML_GradientBoost, 
                          #alg.ML_RandomForest,
                          #alg.ML_SVM,
                          #alg.ML_LinearRegression,
                          #alg.ML_RidgeRegression,
                          #alg.ML_LassoRegression,
                          #alg.ML_ElasticNet,
                          #alg.ML_BasianRegression,
                          #alg.ML_ARDRegression,
                          #alg.ML_kNN,
                          #alg.ML_DNN,
                          #alg.ML_CNN,
                          #alg.ML_XGBoost,
                          #alg.ML_LightGBM,
                          #alg.ML_HistGradientBoost,
                          alg.ML_LSTM,
                          #alg.ML_RNN,
                          #alg.ML_GRU
                          ]
        #if not config.is_regression:
        #    algorithm_list.append(alg.ML_NaiveBayes)

        #algorithm_list = [alg.ML_HistGradientBoost]

        predict_result_df = pd.DataFrame()
        training_result_df = pd.DataFrame()
        importance_df = pd.DataFrame()
        
        start_date = date(2004,1,1)
        value_date = date(2016,1,1)
        end_date = date(2019,5,31)
        for algo in algorithm_list:
            ml_algo = algo(IsRegression=is_regression, with_grid_cv=config.with_grid_cv)
            feature_manager = FeatureVectorManager(FilePath=input_file_name, 
                                                   TrainingStartDate=start_date,
                                                   PredictStartDate=value_date,
                                                   PredictEndDate=end_date,
                                                   IsRegression=is_regression,
                                                   ExecPCA=exec_pca,
                                                   MaxLen=None if 'ml_time_series' not in ml_algo.__module__  and 'ml_cnn' not in ml_algo.__module__ else ml_algo.maxlen)
            
            ml_algo.learn(training_data=feature_manager.training_data,
                          training_label=feature_manager.training_label,
                          tunes_param=config.parameter_tuning)
            
            training_result_df = training_result_df.append(
                                        calc_training_result(ml_algo,
                                                             feature_manager.training_data,
                                                             feature_manager.training_label,
                                                             value_date))
            if 'ml_time_series' not in ml_algo.__module__:
                predicted = ml_algo.predict(feature_manager.predict_data)
                predict_result = pd.DataFrame(predicted,
                                              index=feature_manager.predict_data.index[-len(predicted):],
                                              columns=['Predict'])
                
            else:
                
                ts_index = feature_manager.training_data.index[-(feature_manager.predict_data.shape[0]-ml_algo.maxlen):]
                import pdb;pdb.set_trace()
                predict_result = pd.DataFrame(ml_algo.predict(feature_manager.predict_data),
                                              index=ts_index,
                                              columns=['Predict'])
            predict_result['Algorithm'] = ml_algo.__class__.__name__
            predict_result_df = predict_result_df.append(predict_result)

            if ml_algo.__class__.__name__ in config.importance_models:
                importance_df = importance_df.append(create_importance(ml_algo.__class__.__name__,
                                                                        ml_algo.importance,
                                                                        feature_manager.training_data.columns,
                                                                        value_date))

            #Post Process for each week
            feature_manager.dispose()
            ml_algo.dispose()
            del ml_algo
            del feature_manager
            gc.collect()

            
        #Result Output Process
        predict_result_df.index.name='ValueDate'
        predict_result_df.to_csv('./output/predict_result_{0}_{1}_{2}_{3}.csv'\
                                .format('PCA' if exec_pca else 'NoPCA',
                                        int(training_month),
                                        'Reg' if is_regression else 'Class',
                                        output_suffix), 
                                 index=True)
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
                                       PredictedLabel=create_label(input_file_name))
        #import pdb;pdb.set_trace()
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
