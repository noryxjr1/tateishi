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
#from keras import losses
import tensorflow as tf
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.utils import np_utils

from algorithm.ml_base import ML_Base

class ML_DNN(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._out_dim1 = kwargs.get('out_dim1',60)
        self._out_dim2 = kwargs.get('out_dim2',40)
        self._out_dim3 = kwargs.get('out_dim3',30)
        self._optimizer = kwargs.get('optimizer','adadelta')
        self._nb_epoch = kwargs.get('nb_epoch',1000)
        self._dropout1 = kwargs.get('dropout1',0.5)
        self._dropout2 = kwargs.get('dropout2',0.5)
        self._dropout3 = kwargs.get('dropout3',0.5)
        self._batch_size = kwargs.get('batch_size',100)
        self._activation = kwargs.get('activation', 'relu')


        

        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def learn(self, training_data, training_label, tunes_param=False):
        #config.gpu_options.allow_growth = True
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", # specify GPU number
                                                          allow_growth=True))
        
        sess = tf.Session(config=config)
        K.set_session(sess)

        seed = 7
        self._input_dim = training_data.shape[1]
        np.random.seed(seed)
        
        # evaluate model with standardized dataset
        if self._is_regression:
            self._model = KerasRegressor(build_fn=self._create_model,
                                         epochs=self._nb_epoch,
                                         batch_size=self._batch_size,
                                         #validation_data=(np.array(test_data),
                                         #np.array(test_label)),
                                         verbose=0)

            hist = self._model.fit(np.array(training_data),
                                   np.array(training_label),
                                   callbacks=[EarlyStopping(monitor='loss',
                                                            patience=100,
                                                            verbose=0)]
                                   , validation_split = 0.2
                                #validation_data=(np.array(training_data),
                                #                 np.array(training_label))
                                    )
            #import pdb;pdb.set_trace()
        else:
            self._model = self._create_model()
            hist = self._model.fit(np.array(training_data)
                                   , self._encode_one_hot(training_label)
                                   , callbacks=[EarlyStopping(monitor='loss'
                                                              ,patience=10000
                                                              ,verbose=0)]
                                   , batch_size=self._batch_size
                                   , nb_epoch=self._nb_epoch
                                   , validation_split = 0.2
                                   #, validation_data=(np.array(test_data)
                                   #                   ,
                                   #                   self._change_label_format(np.array(test_label)))
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
            return 1 if predicted[0][0] > predicted[0][1] else 0
            #return self._model.predict_classes(test_data)

    def predict(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)
        if self._is_regression:
            return super().predict(test_data)
        else:
            predicted = self._model.predict(test_data)
            return [1 if predicted[i][0] > predicted[i][1] else 0 \
                    for i in range(len(predicted))]

    def _encode_one_hot(self, label):
        return np.array([[1,0] if label.Return.iloc[i] > 0.0 else [0,1] \
                        for i in range(label.shape[0])])

        
    #def _weight_variable(self, shape):
    #    return K.truncated_normal(shape, stddev=0.01)

    def _create_model(self):
        if self._is_regression:
            loss_func = 'mean_squared_error'
        else:
            loss_func = 'categorical_crossentropy'
        model = Sequential()
        model.add(Dense(self._out_dim1, 
                        input_dim=self._input_dim,
                        kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(self._activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(self._dropout1))
        model.add(Dense(self._out_dim2, kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(self._activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(self._dropout2))
        model.add(Dense(self._out_dim3, kernel_initializer=TruncatedNormal(stddev=0.01)))
        
        model.add(Activation(self._activation))
        #model.add(Activation(PReLU()))
        
        #model.add(BatchNormalization())
        model.add(Dropout(self._dropout3))


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
        return np.matrix([[1,0] if label_data[i] == 0 else [0,1] \
                         for i in range(len(label_data))])

    def dispose(self):
        super().dispose()
        K.clear_session()



