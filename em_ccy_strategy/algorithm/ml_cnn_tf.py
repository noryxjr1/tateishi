# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


from algorithm.ml_base import ML_Base
#from tuning.ml_cv_search import DNN_SearchCV

class ML_CNN_TF(ML_Base):
    cv_model = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._maxlen = kwargs.get('maxlen', 5)
        self._nb_epoch = kwargs.get('nb_epoch',150)
        self._batch_size = kwargs.get('batch_size',100)
        self._with_functional_api = kwargs.get('with_functional_api', True)
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

        if tunes_param:
            self._params = self.tune_param(training_data, training_label, self._is_regression)
        training_data = self._create_ts_data(training_data)
        
        self._model = self._create_model(data_shape=training_data.shape, **self._params)
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
            return 0 if predicted[0][0] > predicted[0][1] else 1

    def predict(self, test_data):
        test_data = self._create_ts_data(test_data)
        if type(test_data) != np.array:
            test_data = np.array(test_data)
        #import pdb;pdb.set_trace()
        if self._is_regression:
            return self._model.predict(test_data)
        else:
            pred_result = self._model.predict(test_data)
            return [0 if pred_result[i][0] > pred_result[i][1] else 1 for i in range(len(pred_result))]
            

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
            loss_func = 'sparse_categorical_crossentropy'
        
        conv1d_1 = Conv1D(filters=1
                          , kernel_size=channel_size
                          , padding='valid'
                          , activation=activation
                          , input_shape=(data_shape[1], data_shape[2]))
        conv1d_2 = Conv1D(filters=1
                          , kernel_size=5
                          , padding='same'
                          , activation='tanh')
        global_max_1d = GlobalMaxPooling1D()

        if self._with_functional_api:
            inputs = Input(name='layer_in', shape=(data_shape[1], data_shape[2]))
            x1 = conv1d_1(inputs)
            x2 = conv1d_2(x1)
            x3 = global_max_1d(x2)
            if not self._is_regression:
                layer_out = Dense(name='layer_out', units=2)
                acti_out = Activation('softmax', name='acti_out')
                outputs = acti_out(layer_out(x3))
            else:
                outputs = x3

            model = Model(inputs=inputs, outputs=outputs, name='cnn_model_constructor')
        else:
            model = Sequential([conv1d_1, conv1d_2, global_max_1d], 
                               name='cnn_seq_constructor')
            if not self._is_regression:
                model.add(Dense(name='layer_out', units=2))
                model.add(Activation('softmax', name='acti_out'))
        
        model.compile(optimizer='adam',
                        loss=loss_func,
                        metrics=['acc'])

        return model


    def dispose(self):
        super().dispose()
        backend.clear_session()
