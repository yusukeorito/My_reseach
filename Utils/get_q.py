#overlapの計算に用いる関数
import numpy as np
import pickle
import joblib
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import Callback,ModelCheckpoint

class CustomConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, const):
        self.mask = mask
        self.const = const

    def __call__(self, w):
        # マスク行列を使用して、指定された部分を0で固定する
        w.assign(tf.math.multiply(w, self.mask) + self.const)
        return w

class DenseCoupleConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, w):
        # マスク行列を使用して、指定された部分を0で固定する
        w.assign(tf.math.multiply(w, self.mask))
        return w 
 
 
      
def get_mask(shape, C, dtype=int):
    masks = np.zeros(shape)
    consts = np.random.normal(size=shape)
    for col in range(shape[1]):
        non_zero_indices = np.random.choice(shape[0], C, replace=False)
        masks[non_zero_indices, col] = 1
        consts[non_zero_indices, col] = 0
    masks = tf.constant(masks, dtype=tf.float32)
    consts = tf.constant(consts, dtype=tf.float32)
    return masks, consts



class LogEpochIntermediateCallcack(Callback):
  def __init__(self, layer_name_list,CFG,X_train,path):
    self.layer_name_list = layer_name_list
    self.spin_dict = {key: [] for key in self.layer_name_list}
    #self.spin_dict_test = {key: [] for key in self.layer_name_list}
    self.nextMeas=1
    self.CFG = CFG
    self.X_train = X_train
    self.path = path
    if self.CFG['M'] == 60000:
      idx = np.random.choice(X_train.shape[0], size=6000, replace=False)
      self.X_train_ = X_train[idx]
      print(self.X_train_.shape)
    


  def on_train_begin(self, batch, logs=None):
    self.spin_dict['time'] = [0]
    #self.spin_dict_test['time'] = [0]
    for l in self.layer_name_list:
      intermediate_layer_model = tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer(l).output)
      if self.CFG['M'] == 60000:
        intermediate_output = intermediate_layer_model.predict(self.X_train_)
      else:
        intermediate_output = intermediate_layer_model.predict(self.X_train)
      #intermediate_output_test = intermediate_layer_model.predict(test_in)
      tf.keras.backend.clear_session()
      self.spin_dict[l].append(intermediate_output)
      #self.spin_dict_test[l].append(intermediate_output_test)

  def on_epoch_begin(self, epoch, logs=None):
      self.ep = epoch

  def on_epoch_end(self, batch, logs):
      if self.ep+1==self.nextMeas:
        for l in self.layer_name_list:
          intermediate_layer_model = tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer(l).output)
          if self.CFG['M'] == 60000:
            intermediate_output = intermediate_layer_model.predict(self.X_train_)
          else:
            intermediate_output = intermediate_layer_model.predict(self.X_train)
          #intermediate_output_test = intermediate_layer_model.predict(test_in)
          tf.keras.backend.clear_session()
          self.spin_dict[l].append(intermediate_output)
          #self.spin_dict_test[l].append(intermediate_output_test)
        self.spin_dict['time']+=[self.ep+1]
        #self.spin_dict_test['time']+=[self.ep+1]
        self.nextMeas=int(self.nextMeas*1.1)
        if self.ep+1==self.nextMeas:
          self.nextMeas = self.nextMeas+1


  def on_train_end(self, logs=None):
    for l in self.layer_name_list:
      self.spin_dict[l] = np.array(self.spin_dict[l])
      #self.spin_dict_test[l] = np.array(self.spin_dict_test[l])
    joblib.dump(self.spin_dict,self.path)
    #with open(f'./Output/Spin/M600/model{self.model_num}_M{CFG.M}_L{CFG.L}_A{CFG.A}_test.txt','wb') as handle:
        #pickle.dump(self.spin_dict_test, handle)



def get_normalized_spin(SpinA, SpinB):
  spinA_norm = SpinA.copy()
  spinB_norm = SpinB.copy()
  for l in tqdm(CFG.layer_name_list):
        squared_sum_A = np.sum(SpinA[l]**2, axis=2)
        squared_sum_B = np.sum(SpinB[l]**2, axis=2)
        # 規格化定数を計算
        normalization_constA = np.sqrt(100 / squared_sum_A)
        normalization_constB = np.sqrt(100 / squared_sum_B)
        # 規格化した配列を計算
        spinA_norm[l] = SpinA[l] * normalization_constA[:, :, np.newaxis]
        spinB_norm[l] = SpinB[l] * normalization_constB[:, :, np.newaxis]

  return spinA_norm, spinB_norm


def calc_q_(A: np.ndarray, B: np.ndarray) -> float:
    M, N = A.shape
    dot_product = np.dot(A.T, B)
    x = np.sum(dot_product ** 2)
    x /= N * M * M
    x -= N / M
    return x

def calc_sim_q(A: np.ndarray, B: np.ndarray) -> float:
  mean = A * B
  sim_q = np.mean(mean)
  return sim_q


def get_sim_q(spinA, spinB, layer_name_list, path):
  qab_dict={'time':spinA['time']}#時刻の初期化
  qaa_dict={'time':spinA['time']}
  sim_q_dict={'time':spinA['time']}
  for l in tqdm(layer_name_list):
      qab_list=[]
      qaa_list=[]
      sim_q_list=[]
      for i in range(len(spinA[l])):
          ab = calc_sim_q(spinA[l][i],spinB[l][i])
          aa = calc_sim_q(spinA[l][i],spinA[l][i])
          bb = calc_sim_q(spinB[l][i],spinB[l][i])
          sim_q = ab/(np.sqrt(aa)*np.sqrt(bb))
          qab_list.append(ab)
          qaa_list.append(aa)
          sim_q_list.append(sim_q)
      qab_dict[l] = qab_list
      qaa_dict[l] = qaa_list
      sim_q_dict[l] = sim_q_list
  joblib.dump(sim_q_dict, path)
  return sim_q_dict


def get_q2(spinA, spinB, layer_name_list ,path):
    qab_dict={'time':spinA['time']}#時刻の初期化
    qaa_dict={'time':spinA['time']}
    q2_dict={'time':spinA['time']}
    for l in tqdm(layer_name_list):
        qab_list=[]
        qaa_list=[]
        q2_list=[]
        for i in range(len(spinA[l])):
            ab = calc_q_(spinA[l][i],spinB[l][i])
            aa= calc_q_(spinA[l][i],spinA[l][i])
            bb = calc_q_(spinB[l][i],spinB[l][i])
            q2 = ab/(np.sqrt(aa)*np.sqrt(bb))
            qab_list.append(ab)
            qaa_list.append(aa)
            q2_list.append(q2)
        qab_dict[l] = qab_list
        qaa_dict[l] = qaa_list
        q2_dict[l] = q2_list
    """
    with open(f'./Output/Overlap/q/{CFG.alg}_qab_norm_M{CFG.M}_L{CFG.L}_A{CFG.A}_{CFG.train}.txt','wb') as handle:
        pickle.dump(qab_dict, handle)
    with open(f'./Output/Overlap/q/{CFG.alg}_qaa_norm_M{CFG.M}_L{CFG.L}_A{CFG.A}_{CFG.train}.txt','wb') as handle:
        pickle.dump(qaa_dict, handle)
    """
    joblib.dump(q2_dict, path)
    return qab_dict, qaa_dict, q2_dict

def get_layer_overlap(qab:dict,qaa:dict,q2:dict,sim_q:dict, layer_name_list, path):
  layer_dict={}
  layer_q2=[]
  layer_qab=[]
  layer_qaa=[]
  layer_sim_q=[]

  for i, l in enumerate(layer_name_list):
      layer_q2.append(q2[l][-1])#平衡状態のOverlapを取得
      layer_qab.append(qab[l][-1])
      layer_qaa.append(qaa[l][-1])
      layer_sim_q.append(sim_q[l][-1])
  layer_dict['q2']=layer_q2
  layer_dict['qab']=layer_qab
  layer_dict['qaa']=layer_qaa
  layer_dict['sim_q']=layer_sim_q

  joblib.dump(layer_dict, path)
  return layer_dict