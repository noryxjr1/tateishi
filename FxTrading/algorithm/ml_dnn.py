# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.layers.advanced_activations import PReLU, LeakyReLU
#from keras import losses
import tensorflow as tf
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.utils import np_utils

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import DNN_SearchCV
from util.ml_config_parser import MLConfigParser

class ML_DNN(ML_Base):
    cv_model = DNN_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._config = MLConfigParser()
        self._nb_epoch = kwargs.get('nb_epoch',150)
        self._batch_size = kwargs.get('batch_size',100)
        self._params = {'out_dim1': kwargs.get('out_dim1',40),
                        'out_dim2': kwargs.get('out_dim2',80),
                        'out_dim3': kwargs.get('out_dim3',10),
                        'optimizer': kwargs.get('optimizer','adam'),
                        #'nb_epoch': kwargs.get('nb_epoch',100),
                        'dropout1': kwargs.get('dropout1',0.7),
                        'dropout2': kwargs.get('dropout2',0.8),
                        'dropout3': kwargs.get('dropout3',0.7),
                        #'batch_size': kwargs.get('batch_size',100),
                        'activation': kwargs.get('activation','relu'),
                        #'activation': kwargs.get('activation','sigmoid'),
                        }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        import os
        seed = 1234
        np.random.seed(seed)
        if self._config.parameter_tuning:
            
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", # specify GPU number
                                                          allow_growth=True))
        sess = tf.Session(config=config)
        K.set_session(sess)

        #if tunes_param:
        #    self._params = self.tune_param(training_data, training_label, self._is_regression)
        
        model_file_path = '{model_name}_{value_date}.h5'.format(model_name=self.__class__.__name__,
                                                                value_date=training_data.index[-1].strftime('%Y%m%d'))
        model_file_path = os.path.join('output', 'model', model_file_path)
        # evaluate model with standardized dataset
        if self._is_regression:
            
            self._model = KerasRegressor(build_fn=self._create_model,
                                         input_dim=training_data.shape[1],
                                         verbose=1,
                                         **self._params)
            hist = self._model.fit(np.array(training_data)
                                   ,np.array(training_label)
                                   , callbacks=[EarlyStopping(monitor='loss',
                                                              patience=100000,
                                                              verbose=0),
                                                #ModelCheckpoint(model_file_path, 
                                                #                save_best_only=True),
                                                #TensorBoard(log_dir='logs')
                                                ]
                                   , batch_size=self._batch_size
                                   , epochs=self._nb_epoch
                                   , validation_split = 0.2)
        else:
            
            self._model = KerasClassifier(build_fn=self._create_model,
                                          input_dim=training_data.shape[1],
                                          verbose=0,
                                          **self._params, 
                                          validation_split = 0.2)
            #self._params['input_dim'] = training_data.shape[1]
            #self._model = self._create_model(**self._params)
            hist = self._model.fit(np.array(training_data)
                                   , self._encode_one_hot(training_label)
                                   , callbacks=[EarlyStopping(monitor='loss'
                                                              ,patience=100000
                                                              ,verbose=0),
                                                #ModelCheckpoint(model_file_path, 
                                                #                save_best_only=True),
                                                #TensorBoard(log_dir='logs')
                                                ]
                                   , batch_size=self._batch_size
                                   , epochs=self._nb_epoch
                                   , validation_split = 0.2
                                   )
        #import matplotlib.pyplot as plt
        #plt.plot(hist.history['loss'])
        #import pdb;pdb.set_trace()
            

    def predict_one(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)
        if self._is_regression:
            return float(self._model.predict(test_data))
        else:
            return self._model.predict(test_data)[0]
            #return 1 if predicted[0][0] > predicted[0][1] else 0

    def predict(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)

        if self._is_regression:
            return super().predict(test_data)
        else:
            return self._model.predict(test_data)
            

    def _encode_one_hot(self, label):
        return np.array([[0,1] if label.Return.iloc[i] > 0.0 else [1,0] \
                        for i in range(label.shape[0])])

        
    #def _weight_variable(self, shape):
    #    return K.truncated_normal(shape, stddev=0.01)

    def _create_model(self, 
                      input_dim, 
                      out_dim1,
                      out_dim2,  
                      out_dim3,  
                      optimizer, 
                      dropout1,  
                      dropout2,  
                      dropout3,  
                      activation='relu'):
        if self._is_regression:
            loss_func = 'mean_squared_error'
        else:
            loss_func = 'categorical_crossentropy'
        model = Sequential()
        model.add(Dense(out_dim1, 
                        input_dim=input_dim,
                        kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(dropout1))
        model.add(Dense(out_dim2, kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(dropout2))
        model.add(Dense(out_dim3, kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(dropout3))


        if self._is_regression:
            model.add(Dense(1))
        else:
            model.add(Dense(2))
            model.add(Activation('softmax'))
        
        model.compile(loss=loss_func
                      , optimizer='adadelta'
                      , metrics=['accuracy'])
        return model


    def _change_label_format(self, label_data):
        return np.matrix([[1,0] if label_data[i] == 0 else [0,1]
                         for i in range(len(label_data))])

    def dispose(self):
        super().dispose()
        K.clear_session()


