# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU,SimpleRNN
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from algorithm.ml_base import ML_Base

class ML_TimeSeries_TF(ML_Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._out_dim1 = kwargs.get('out_dim1',30)
        self._nb_epoch = kwargs.get('nb_epoch',150)
        self._batch_size = kwargs.get('batch_size',100)
        self._with_functional_api = kwargs.get('with_functional_api', True)
        self._params = {'out_dim1': kwargs.get('out_dim1',60),
                        'nb_epoch': kwargs.get('nb_epoch',1000),
                        'batch_size': kwargs.get('batch_size',100)}

        self._maxlen = kwargs.get('maxlen', 3)

        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    @property
    def maxlen(self):
        return self._maxlen
    

    def _create_ts_data(self, target_data):
        ts_data = []
        for i in range(self._maxlen, len(target_data)+1):
            ts_data.append(np.array(target_data.iloc[i-self._maxlen:i]))

        return np.array(ts_data)


    def learn(self, training_data, training_label, tunes_param=False):
        seed = 1234
        self._input_dim = training_data.shape[1]
        np.random.seed(seed)
        
        ts_training_data = self._create_ts_data(training_data)
        ts_training_label = self._create_ts_data(training_label)

        self._model = self._create_model(input_dim=training_data.shape[1],
                                             out_dim1=self._params['out_dim1'])
        hist = self._model.fit(ts_training_data,
                                ts_training_label
                                , callbacks=[EarlyStopping(monitor='loss'
                                                            ,patience=1
                                                            ,verbose=0)]
                                , batch_size=self._batch_size
                                , nb_epoch=self._nb_epoch
                                , validation_split = 0.2
                                )

    def predict_one(self, test_data):
        if self._is_regression:
            return float(self._model.predict(test_data)[0][-1])
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
            
            return [1 if predicted[i][-1] > 0.5 else 0
                    for i in range(len(predicted))]


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
        backend.clear_session()


class ML_LSTM_TF(ML_TimeSeries_TF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def _create_model(self, input_dim, out_dim1):
        lstm = LSTM(batch_input_shape=(None, self._maxlen, input_dim),
                    name='lstm1',
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    units=out_dim1,
                    return_sequences=True,
                    activation='linear')
        layer1 = Dense(name='layer1', units=1,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros',
                       activation='linear' if self._is_regression else None)
        if self._with_functional_api:

            inputs = Input(name='layer_in', shape=(self._maxlen, input_dim))
            
            outputs = layer1(lstm(inputs))
            model = Model(inputs=inputs, outputs=outputs, name='lstm_constructor')

        else:
            model = Sequential([lstm, layer1], name='lstm_constructor')
       
        model.compile(loss='mean_squared_error'
                      , optimizer='adam'
                      , metrics=['accuracy'])
        return model

class ML_RNN_TF(ML_TimeSeries_TF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def _create_model(self, input_dim, out_dim1):
        rnn = SimpleRNN(batch_input_shape=(None, self._maxlen, input_dim),
                        name='rnn1',
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        units=out_dim1,
                        return_sequences=True,
                        activation='linear')
        layer1 = Dense(name='layer1', units=1,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activation='linear' if self._is_regression else None)
        if self._with_functional_api:
            inputs = Input(name='layer_in', shape=(self._maxlen, input_dim))
            outputs = layer1(rnn(inputs))
            model = Model(inputs=inputs, outputs=outputs, name='rnn_constructor')

        else:
            model = Sequential([rnn, layer1], name='rnn_constructor')
       
        
        model.compile(loss='mean_squared_error'
                      , optimizer='adam'
                      , metrics=['accuracy'])
        return model


class ML_GRU_TF(ML_TimeSeries_TF):

    def __init__(self, **kwargs):
        kwargs['nb_epoch'] = 100
        super().__init__(**kwargs)
        
    def _create_model(self, input_dim, out_dim1):
        
        gru = GRU(batch_input_shape=(None, self._maxlen, input_dim),
                  name='gru1',
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  units=out_dim1,
                  return_sequences=True,
                  activation='linear')
        layer1 = Dense(name='layer1', units=1,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activation='linear' if self._is_regression else None)
        if self._with_functional_api:
            inputs = Input(name='layer_in', shape=(self._maxlen, input_dim))
            model = Model(inputs=inputs, 
                          outputs=layer1(gru(inputs)), name='gru_constructor')

        else:
            model = Sequential([gru, layer1], name='gru_constructor')
       
        
        model.compile(loss='mean_squared_error'
                      , optimizer='adam'
                      , metrics=['accuracy'])
        return model
