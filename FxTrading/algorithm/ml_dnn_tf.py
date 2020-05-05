# -*- coding: utf-8 -*-
"""
Created on Mon May 4 04:00:00 2020
@author: nory.xjr1
"""
import tensorflow as tf
if tf.__version__ < '2.0.0': raise Exception('Tensorflow version must be more than 2.0.0')

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from algorithm.ml_base import ML_Base
from tuning.ml_cv_search import DNN_SearchCV
from util.ml_config_parser import MLConfigParser


class ML_DNN_TF(ML_Base):
    cv_model = DNN_SearchCV
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._config = MLConfigParser()
        self._nb_epoch = kwargs.get('nb_epoch', 150)
        self._batch_size = kwargs.get('batch_size',100)
        self._with_functional_api = kwargs.get('with_functional_api', False)

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
                        }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        
        seed = 1234
        np.random.seed(seed)
        if self._config.parameter_tuning:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        #config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(visible_device_list="0", # specify GPU number
        #                                                  allow_growth=True))
        #sess = tf.compat.v1.Session(config=config)
        #K.set_session(sess)

        #if tunes_param:
        #    self._params = self.tune_param(training_data, training_label, self._is_regression)
        
        model_file_path = '{model_name}_{value_date}.h5'.format(model_name=self.__class__.__name__,
                                                                value_date=training_data.index[-1].strftime('%Y%m%d'))
        model_file_path = os.path.join('output', 'model', model_file_path)

        
        self._model = self._create_model(input_dim=training_data.shape[1], **self._params)

        hist = self._model.fit(np.array(training_data)
                                , np.array(training_label)
                                , callbacks=[EarlyStopping(monitor='loss'
                                                            ,patience=100
                                                            ,verbose=0),
                                            #ModelCheckpoint(model_file_path, 
                                            #                save_best_only=True),
                                            #TensorBoard(log_dir='logs')
                                            ]
                                , batch_size=self._batch_size
                                , epochs=self._nb_epoch
                                , validation_split = 0.2
                                , verbose=2
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
            return 0 if predicted[0][0] > predicted[0][1] else 1

    def predict(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)

        if self._is_regression:
            return super().predict(test_data)
        else:
            pred_result = self._model.predict(test_data)
            return [0 if pred_result[i][0] > pred_result[i][1] else 1 for i in range(len(pred_result))]

    
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
            last_output_dim = 1
        else:
            loss_func = 'sparse_categorical_crossentropy'
            last_output_dim = 2

        activation1 = Activation(activation, name='activation1')
        activation2 = Activation(activation, name='activation2')
        activation3 = Activation(activation, name='activation3')
        do1 = Dropout(dropout1, name='dropout1')
        do2 = Dropout(dropout2, name='dropout2')
        do3 = Dropout(dropout3, name='dropout3')
        
        layer1 = Dense(name='layer1', units=out_dim1,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros')
        layer2 = Dense(name='layer2', units=out_dim2,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros')
        layer3 = Dense(name='layer3', units=out_dim3,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros')
        layer_out = Dense(name='layer_out', units=last_output_dim)
        acti_out = Activation('softmax', name='acti_out')

        if self._with_functional_api:
            inputs = Input(name='layer_in', shape=(input_dim,))
            x1 = do1(activation1(layer1(inputs)))
            x2 = do2(activation2(layer2(x1)))
            x3 = do3(activation3(layer3(x2)))
            outputs = acti_out(layer_out(x3))

            model = Model(inputs=inputs, outputs=outputs, name='dnn_model_constructor')
        else:
            model = Sequential([layer1, activation1, do1, 
                                layer2, activation2, do2, 
                                layer3, activation3, do3, 
                                layer_ou,], 
                               name='dnn_seq_constructor')

        model.compile(optimizer=optimizer,
                        loss=loss_func,
                        metrics=['acc'])

        return model


    #def _change_label_format(self, label_data):
    #    return np.matrix([[1,0] if label_data[i] == 0 else [0,1]
    #                     for i in range(len(label_data))])

    def dispose(self):
        super().dispose()
        backend.clear_session()


