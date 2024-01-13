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


def create_dense_coupling_model(params:dict, W, B, Mask_list,Const_list):
    model=Sequential(name='sparse_connect_model')
    model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,input_shape=(params['N'],), name='Affine1',kernel_constraint=DenseCoupleConstraint(mask=Mask_list[0])))
    model.add(BatchNormalization(name='batch_normalization1'))
    model.add(Activation('relu', name='activation1'))
    for i in range(1, params['num_layers']-1):
        model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,name=f'Affine{i+1}',kernel_constraint=DenseCoupleConstraint(mask=Mask_list[i])))
        model.add(BatchNormalization(name=f'batch_normalization{i+1}'))
        model.add(Activation(params['activation'],name=f'activation{i+1}'))
        
    model.add(Dense(params['num_classes'],kernel_initializer=W,bias_initializer=B,name='output',kernel_constraint=DenseCoupleConstraint(mask=Mask_list[params["num_layers"]-1])))
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
        
    model.add(Dense(params['num_classes'],kernel_initializer=W,bias_initializer=B,name='output',kernel_constraint=CustomConstraint(mask=Mask_list[params["num_layers"]-1],const=Const_list[params["num_layers"]-1])))
    model.add(BatchNormalization(name=f'batch_normalization{params["num_layers"]}'))
    model.add(Activation('softmax', name=f'activation{params["num_layers"]}'))
    model.compile(loss=params['loss'], optimizer=params['optimizer'],metrics=[params['metric']])
    return model

def create_fully_AEmodel(params:dict, W, B):
    model = Sequential(name='AE_fully_model')
    model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,input_shape=(params['N'],), name='Affine1',))
    model.add(BatchNormalization(name='batch_normalization1'))
    model.add(Activation('relu', name='activation1'))
    for i in range(1, params['num_layers']-1):
        model.add(Dense(params['N'], kernel_initializer=W,bias_initializer=B,name=f'Affine{i+1}'))
        model.add(BatchNormalization(name=f'batch_normalization{i+1}'))
        model.add(Activation(params['activation'],name=f'activation{i+1}'))

    model.add(Dense(params['num_classes'],kernel_initializer=W,bias_initializer=B,name='output',))
    model.add(BatchNormalization(name=f'batch_normalization{params["num_layers"]}'))
    model.add(Activation('relu', name=f'activation{params["num_layers"]}'))
    model.compile(loss=params['loss'], optimizer='adam')
    return model