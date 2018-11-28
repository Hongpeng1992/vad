import tensorflow as tf
import numpy as np

class CNNModel:
    def __init__(self, hparams):
        self._hparams = hparams
        
        mean_std = np.load(hparams.mean_std_path)
        
        self._mean = tf.constant(mean_std['mean'], dtype=tf.float32, name='mean')
        self._std = tf.constant(mean_std['std'], dtype=tf.float32, name='std')
        
        
    def initialize(self, features, labels=None, is_training=False):
        self.features = features
        self.labels = labels
        self.is_training = is_training
        self.nodes = []
    
#         if self._hparams.normalize_features:
#             with tf.name_scope('normalization'):
#                 nnet = self.features - self._mean
#                 nnet /= self._std
#         else:
        nnet = tf.identity(self.features)
            
        with tf.variable_scope('CNN'):
            nnet = tf.layers.conv1d(nnet, self._hparams.n_hidden, kernel_size=1, activation=tf.nn.relu, name='conv1D_1') # [n_cand, n_hidden]
            self.nodes.append(nnet)
            nnet = tf.layers.conv1d(nnet, 1, kernel_size=1, activation=tf.nn.sigmoid, name='conv2d_2')
            self.nodes.append(nnet)
            nnet = tf.squeeze(nnet)
            
            self.predictions = tf.reduce_max(nnet, axis=1)
        
    
    def compute_loss(self):
        with tf.name_scope('loss'):
            cost = -self.labels * tf.log(self.predictions + self._hparams.eps)
            self.loss = tf.reduce_mean(cost)

    
    def compute_gradients(self):
        with tf.name_scope('gradients'):
            vars = tf.trainable_variables()
            gradients = tf.gradients(self.loss, vars)
            self.gradvars = list(zip(gradients, vars))
