# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.recurrent import LSTM, SimpleRNN, GRU

import tensorflow as tf
from keras import backend as K
from keras.initializers import TruncatedNormal

from algorithm.ml_base import ML_Base

class ML_TimeSeries(ML_Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._out_dim1 = kwargs.get('out_dim1',30)
        self._nb_epoch = kwargs.get('nb_epoch',150)
        self._batch_size = kwargs.get('batch_size',100)
        self._params = {'out_dim1': kwargs.get('out_dim1',60),
                        'nb_epoch': kwargs.get('nb_epoch',1000),
                        'batch_size': kwargs.get('batch_size',100)}

        self._maxlen = kwargs.get('maxlen', 3)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",
                                                          allow_growth=True))
        
        sess = tf.Session(config=config)
        K.set_session(sess)

        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def maxlen(self):
        return self._maxlen
    

    def _create_ts_data(self, target_data):
        ts_data = []
        for i in range(self._maxlen, len(target_data)+1):
            ts_data.append(np.array(target_data.iloc[i-self._maxlen:i]))

        #for i in range(len(target_data)-self._maxlen):
        #    ts_data.append(np.array(target_data.iloc[i:i+self._maxlen]))
        return np.array(ts_data)


    def learn(self, training_data, training_label, tunes_param=False):
        seed = 1234
        self._input_dim = training_data.shape[1]
        np.random.seed(seed)
        
        ts_training_data = self._create_ts_data(training_data)
        ts_training_label = self._create_ts_data(training_label)
        # evaluate model with standardized dataset
        if self._is_regression:
            self._model = KerasRegressor(build_fn=self._create_model,
                                         verbose=1,
                                         input_dim=training_data.shape[1],
                                         **self._params)
            hist = self._model.fit(ts_training_data,
                                   ts_training_label,
                                   callbacks=[EarlyStopping(monitor='loss',
                                                            patience=1,
                                                            verbose=0)]
                                   , batch_size=self._batch_size
                                   , epochs=self._nb_epoch
                                   , validation_split = 0.2
                                   )
        else:
            
            self._model = self._create_model(input_dim=training_data.shape[1],
                                             out_dim1=self._params['out_dim1'])
            hist = self._model.fit(ts_training_data,
                                   ts_training_label
                                   , callbacks=[EarlyStopping(monitor='loss'
                                                              ,patience=1
                                                              ,verbose=0)]
                                   , batch_size=self._batch_size
                                   , nb_epoch=self._nb_epoch
                                   #, validation_split = 0.2
                                   )
        #import matplotlib.pyplot as plt
        #plt.plot(hist.history['loss'])
        #import pdb;pdb.set_trace()
            

    def predict_one(self, test_data):
        if self._is_regression:
            return float(self._model.predict(test_data)[-1])
        else:
            predicted = self._model.predict(test_data)
            return 1 if predicted[0][-1][0] > 0 else 0

    def predict(self, test_data):
        
        if type(test_data) == np.ndarray:
            ts_test_data = test_data
        else:
            ts_test_data = self._create_ts_data(test_data)
        
        if self._is_regression:
            return super().predict(ts_test_data)[:,-1]
        else:
            
            predicted = self._model.predict(ts_test_data)
            #import pdb;pdb.set_trace()
            return [1 if predicted[i][-1] > 0.5 else 0
                    for i in range(len(predicted))]
            #return [1 if predicted[i][0] > predicted[i][1] else 0
            #        for i in range(len(predicted))]

    def predict_one_proba(self, test_data):
        proba = self._model.predict_proba(test_data)[0][-1]
        return [proba, 1-proba]

    def _encode_one_hot(self, label):
        return np.array([[1,0] if label.Return.iloc[i] > 0.0 else [0,1]
                        for i in range(label.shape[0])])

    def _change_label_format(self, label_data):
        return np.matrix([[1,0] if label_data[i] == 0 else [0,1]
                         for i in range(len(label_data))])

    def dispose(self):
        super().dispose()
        K.clear_session()


class ML_LSTM(ML_TimeSeries):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def _create_model(self, input_dim, out_dim1):
        model = Sequential()
        model.add(LSTM(out_dim1, 
                       batch_input_shape=(None, self._maxlen, input_dim),
                       return_sequences=True))

        if self._is_regression:
            model.add(Dense(1))
        else:
            model.add(Dense(1))
            model.add(Activation('linear'))
        
        model.compile(loss='mean_squared_error'
                      , optimizer=Adam(lr=0.001)
                      , metrics=['accuracy'])
        return model

class ML_RNN(ML_TimeSeries):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def _create_model(self, input_dim, out_dim1):
        model = Sequential()
        model.add(SimpleRNN(out_dim1, 
                       batch_input_shape=(None, self._maxlen, input_dim),
                       return_sequences=True))

        if self._is_regression:
            model.add(Dense(1))
        else:
            model.add(Dense(1))
            #model.add(Activation('linear'))
            model.add(Activation('sigmoid'))
        
        model.compile(loss='mean_squared_error'
                      , optimizer=Adam(lr=0.001)
                      , metrics=['accuracy'])
        return model



class ML_GRU(ML_TimeSeries):

    def __init__(self, **kwargs):
        kwargs['nb_epoch'] = 100
        super().__init__(**kwargs)
        
    def _create_model(self, input_dim, out_dim1):
        model = Sequential()
        model.add(GRU(out_dim1, 
                      batch_input_shape=(None, self._maxlen, input_dim),
                      return_sequences=True))

        if self._is_regression:
            model.add(Dense(1))
        else:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        
        model.compile(loss='mean_squared_error'
                      , optimizer=Adam(lr=0.0001)
                      , metrics=['accuracy'])
        return model
