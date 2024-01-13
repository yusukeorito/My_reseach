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

def load_fashion_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
    
    
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


def add_gaussian_noise(image, mean, std_dev, alpha):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + alpha * noise
    return np.clip(noisy_image, 0, 1)  # ピクセル値を0から1の範囲にクリップ


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
    
    
class Layer1WeightsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        layer1_weights = self.model.layers[0].get_weights()[0][:,1]
        print(f"Epoch {epoch + 1} - Layer 1 Weights:")
        print(layer1_weights)