import tensorflow as tf
if tf.__version__ < '2.0.0':
    from .ml_cnn import *
    #from .ml_dnn import *
    from .ml_dnn_fc import *
    from .ml_time_series import *
else:
    from .ml_cnn_tf import ML_CNN_TF as ML_CNN
    from .ml_dnn_tf import ML_DNN_TF as ML_DNN
    from .ml_time_series_tf import ML_RNN_TF as ML_RNN
    from .ml_time_series_tf import ML_LSTM_TF as ML_LSTM
    from .ml_time_series_tf import ML_GRU_TF as ML_GRU


