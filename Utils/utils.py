#汎用的に利用できる関数
import os
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.layers import Dense,BatchNormalization,Activation
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import plot_model


#seedの固定
def set_seed(seed):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def load_mnist_data(mnist_file_path):
    with np.load(mnist_file_path) as data:
        X_train, y_train = data['x_train'], data['y_train']
        X_test, y_test = data['x_test'], data['y_test']
    
    return X_train, y_train, X_test, y_test
    
    
#データの前処理(標準化とデータ型)
def preprocess_data(X_train_, y_train_):
    # Rasterize and normalize samples
    X_train_ = X_train_.reshape(X_train_.shape[0], -1)
    y_train_ = y_train_.reshape(y_train_.shape[0], -1)

    X_train_ = X_train_ / 255
    y_train_ = y_train_ / 255
    # Use 32-bit instead of 64-bit float
    X_train_ = X_train_.astype("float32")
    y_train_ = y_train_.astype("float32")

    return X_train_, y_train_

def create_fully_model(params:dict, W, B):
    model=Sequential(name='fully_connect_model')
    model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,input_shape=(params['N'],), name='Affine1',))
    model.add(BatchNormalization(name='batch_normalization1'))
    model.add(Activation('relu', name='activation1'))
    for i in range(1, params['num_layers']-1):
        model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,name=f'Affine{i+1}'))
        model.add(BatchNormalization(name=f'batch_normalization{i+1}'))
        model.add(Activation(params['activation'],name=f'activation{i+1}'))
        
    model.add(Dense(params['num_classes'],kernel_initializer=W,bias_initializer=B,name='output',))
    model.add(BatchNormalization(name=f'batch_normalization{params["num_layers"]}'))
    model.add(Activation('softmax', name=f'activation{params["num_layers"]}'))
    model.compile(loss=params['loss'], optimizer=params['optimizer'],metrics=[params['metric']])
    return model


def create_sparse_model(params:dict, W, B, Mask_list,Const_list):
    model=Sequential(name='sparse_connect_model')
    model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,input_shape=(params['N'],), name='Affine1',kernel_constraint=CustomConstraint(mask=Mask_list[0],const=Const_list[0])))
    model.add(BatchNormalization(name='batch_normalization1'))
    model.add(Activation('relu', name='activation1'))
    for i in range(1, params['num_layers']-1):
        model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,name=f'Affine{i+1}',kernel_constraint=CustomConstraint(mask=Mask_list[i],const=Const_list[i])))
        model.add(BatchNormalization(name=f'batch_normalization{i+1}'))
        model.add(Activation(params['activation'],name=f'activation{i+1}'))
        
    model.add(Dense(params['num_classes'],kernel_initializer=W,bias_initializer=B,name='output',kernel_constraint=CustomConstraint(mask=Mask_list[9],const=Const_list[9])))
    model.add(BatchNormalization(name=f'batch_normalization{params["num_layers"]}'))
    model.add(Activation('softmax', name=f'activation{params["num_layers"]}'))
    model.compile(loss=params['loss'], optimizer=params['optimizer'],metrics=[params['metric']])
    return model