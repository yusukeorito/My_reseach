#データの読み込みから実行まで全てを実行するくRunnerクラスを定義する
from typing import Any, Dict, Tuple, Union
import sys
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.layers import Dense,BatchNormalization,Activation
from tensorflow.keras.activations import relu


sys.path.append('../../')
from Utils.utils import *
from Utils.get_q import *


class Runner:
    def __init__(self, configs: Dict) -> None:  # type: ignore
        self.exp_name = configs["exp_name"]
        self.run_name = configs["run_name"]
        self.description = configs["description"]
        self.run_id = None
        self.X_train, self.y_train, self.X_test, self.y_test = load_mnist_data(configs["input_path"])
        self.params = configs["params"]
        self.set = configs["setting"]
        self.layer_list = configs["layer_name_list"]
        self.output_path = configs["output_path"]
    

        
    def preprocessing(self) -> np.ndarray:
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        set_seed(self.set['data_seed'])
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        X_train = X_train / 255
        X_test = X_test / 255
        # Use 32-bit instead of 64-bit float
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        
        idx = np.random.choice(X_train.shape[0], size=self.set['M'])
        X_train = X_train[idx]
        y_train = y_train[idx]
        
        X_train_out = X_train
        X_test_out = X_test
        y_train_out = y_train
        y_test_out = y_test
    
        return X_train_out, y_train_out, X_test_out, y_test_out
    

        
    def PCA_SS(self,input_train,input_test):
        # Make an instance of the Model
        pca = PCA(n_components=self.set['N'])
        scaler = StandardScaler()

        train_img = pca.fit_transform(input_train)
        train_img =scaler.fit_transform(train_img)
        
        test_img = pca.transform(input_test)
        test_img =scaler.transform(test_img)
        return train_img, test_img, pca, scaler
    
    def train_model(self, X_train, y_train, X_test, y_test):
        params=self.params
        CFG = self.set
        mask_list = []
        const_list = []
        for i in range(CFG['L']-1):
            mask,const = get_mask(shape=(100,100),C=CFG['C'])
            mask_list.append(mask)
            const_list.append(const)
            
        mask, const = get_mask(shape=(100,10),C=CFG.C)
        mask_list.append(mask)
        const_list.append(const)
            
        
        set_seed(self.set['seed1'])
        w_intializer1 = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
        bias_initializer1 = tf.keras.initializers.Constant(0.1)
        model1 = create_sparse_model(params=params,W=w_intializer1,B=bias_initializer1,Mask_list=mask_list,Const_list=const_list)
        history1 = model1.fit(X_train, y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                verbose=1,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=[LogEpochIntermediateCallcack(layer_name_list=self.layer_list,CFG=self.set,X_train=X_train,
                                                        path=f"../Output/Spin/spinA_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed1']}.pkl")]
                )
        
        joblib.dump(history1.history, f"../Output/Loss/model001_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed1']}.pkl")
        
        
        if self.set['ini_type'] == 'A':
            print('='*15+'ini_type A'+'='*15)
            set_seed(self.set['seed2'])
            w_intializer2 = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
            bias_initializer2 = tf.keras.initializers.Constant(0.1)
        elif self.set['ini_type'] == 'B':
            print('='*15+'ini type B'+'='*15)
            w_intializer2 = w_intializer1
            bias_initializer2 = bias_initializer1
            set_seed(self.set['seed3'])
        
        
        model2 = create_sparse_model(params=params,W=w_intializer2,B=bias_initializer2,Mask_list=mask_list,Const_list=const_list)
        
    
        history2 = model2.fit(X_train, y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                verbose=1,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=[LogEpochIntermediateCallcack(layer_name_list=self.layer_list,CFG=self.set,X_train=X_train,
                                                        path=f"../Output/Spin/spinB_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed2']}.pkl")]
                )
        joblib.dump(history1.history, f"../Output/Loss/model002_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed2']}.pkl")
        
        
        spinA = joblib.load(f"../Output/Spin/spinA_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed1']}.pkl")
        spinB = joblib.load(f"../Output/Spin/spinB_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}_seed{CFG['seed2']}.pkl")
        
        print('save overlap')
        sim_q = get_sim_q(spinA,spinB,layer_name_list=self.layer_list,path=f"../Output/Overlap/time_sim_q_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}.pkl")
        qab, qaa, q2 = get_q2(spinA,spinB,layer_name_list=self.layer_list, path=f"../Output/Overlap/time_q2_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}.pkl")
        layer_q = get_layer_overlap(qab,qaa,q2,sim_q,layer_name_list=self.layer_list, path=f"../Output/Overlap/layer_q_{CFG['data_name']}_ini{CFG['ini_type']}_M{CFG['M']}_L{CFG['L']}.pkl")