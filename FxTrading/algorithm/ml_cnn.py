# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import tensorflow as tf
from keras import backend as K

from algorithm.ml_base import ML_Base
#from tuning.ml_cv_search import DNN_SearchCV

class ML_CNN(ML_Base):
    cv_model = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._maxlen = kwargs.get('maxlen', 5)
        self._nb_epoch = kwargs.get('nb_epoch',150)
        self._batch_size = kwargs.get('batch_size',100)
        self._params = {}
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def maxlen(self):
        return self._maxlen

    def _create_ts_data(self, target_data):
        return np.array([np.array(target_data.iloc[i:i+self._maxlen]) 
                         for i in range(len(target_data)-self._maxlen)])

    def learn(self, training_data, training_label, tunes_param=False):
        seed = 1234
        np.random.seed(seed)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",
                                                          allow_growth=True))
        sess = tf.Session(config=config)
        K.set_session(sess)

        if tunes_param:
            self._params = self.tune_param(training_data, training_label, self._is_regression)
        training_data = self._create_ts_data(training_data)
        
        # evaluate model with standardized dataset
        if self._is_regression:
            
            self._model = KerasRegressor(build_fn=self._create_model,
                                         data_shape=training_data.shape,
                                         verbose=1)
            hist = self._model.fit(np.array(training_data)
                                   ,np.array(training_label.iloc[self._maxlen:])
                                   , callbacks=[EarlyStopping(monitor='val_loss'
                                                              #, min_delta=0
                                                              , patience=1000
                                                              , verbose=1
                                                              , mode='auto')]
                                   , batch_size=self._batch_size
                                   , nb_epoch=self._nb_epoch
                                   , validation_split = 0.2)
        else:
            
            self._model = KerasClassifier(build_fn=self._create_model,
                                          data_shape=training_data.shape,
                                          verbose=1)
            hist = self._model.fit(np.array(training_data)
                                   , self._encode_one_hot(training_label.iloc[self._maxlen:])
                                   , callbacks=[EarlyStopping(monitor='val_loss'
                                                              , patience=1000
                                                              , verbose=1
                                                              , mode='auto')]
                                   , batch_size=self._batch_size
                                   , nb_epoch=self._nb_epoch
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
            predicted = self._model.predict(test_data)
            #return 1 if predicted[0][-1][0] > 0 else 0
            return predicted[0]

    def predict(self, test_data):
        test_data = self._create_ts_data(test_data)
        if type(test_data) != np.array:
            test_data = np.array(test_data)

        return self._model.predict(test_data)
            

    def _encode_one_hot(self, label):
        return np.array([[0,1] if label.Return.iloc[i] > 0.0 else [1,0] \
                        for i in range(label.shape[0])])

   
    def _create_model(self, 
                      data_shape,
                      channel_size = 5,
                      kernel_size = 10,
                      activation='relu'):
        if self._is_regression:
            loss_func = 'mean_squared_error'
        else:
            loss_func = 'categorical_crossentropy'
        
        model = Sequential()
        model.add( Conv1D(filters=channel_size
                          , kernel_size=kernel_size
                          , strides=1
                          , padding="same"
                          , activation=activation
                          , input_shape=(data_shape[1], data_shape[2]),) )
        model.add( Conv1D(filters=1
                          , kernel_size=5
                          , padding='same'
                          , activation='tanh',) )
        model.add( GlobalMaxPooling1D() )

        if not self._is_regression:
            model.add(Dense(2))
            model.add(Activation('sigmoid'))
        model.compile(loss='mse', optimizer='adam')
        
        return model


    #def _change_label_format(self, label_data):
    #    return np.matrix([[1,0] if label_data[i] == 0 else [0,1]
    #                     for i in range(len(label_data))])

    def dispose(self):
        super().dispose()
        K.clear_session()
