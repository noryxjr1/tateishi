# -*- coding: utf-8 -*-
import configparser

class MLConfigParser(object):
    def __init__(self, file_path='./util/Config.ini'):
        self._inifile = configparser.ConfigParser()
        self._inifile.read(file_path)

    ##Base
    @property
    def training_term(self):
        return self._inifile.getint('Base','TrainingTerm')

    @property
    def exec_pca(self):
        return self._inifile.getboolean('Base','ExecPCA')

    @property
    def is_regression(self):
        return self._inifile.getboolean('Base','IsRegression')
    
    @property
    def input_dir(self):
        return self._inifile.get('Base','InputDir')

    @property
    def output_dir(self):
        return self._inifile.get('Base','OutputDir')

    @property
    def feature_file(self):
        return self._inifile.get('Base','FeatureFile')
    
    @property
    def fc_label_file(self):
        return self._inifile.get('Base','FCLabelFile')

    @property
    def feature_list_file(self):
        return self._inifile.get('Base','FeatureListFile')

    @property
    def with_multiprocess(self):
        return self._inifile.getboolean('Base','WithMP')

    @property
    def cpu_count(self):
        return self._inifile.getint('Base','CPUCount')

    @property
    def importance_models(self):
        return self._inifile.get('Base','ImportanceModel').split(',')

    @property
    def fix_start_date(self):
        return self._inifile.getboolean('Base', 'FixStartDate')

    @property
    def parameter_tuning(self):
        return self._inifile.getboolean('Base', 'ParameterTuning')

    @property
    def with_grid_cv(self):
        return self._inifile.getboolean('Base', 'GridCV')
    
    @property
    def scaler_type(self):
        return self._inifile.getint('Base', 'ScalerType')
