import os
import numpy as np
import multiprocessing
import tensorflow as tf

class Dataset:
    def __init__(self, data_folder, hparams):
        self._hparams = hparams
        self._data_folder = data_folder
        
        np.random.seed(self._hparams.random_seed)
        
        filenames = [f for f in os.listdir(self._data_folder) if f.endswith('.npz')]
        np.random.shuffle(filenames)
        
        
        self._train_filenames = filenames[self._hparams.n_valid_files:]
        self._test_filenames = filenames[:self._hparams.n_valid_files]
        
        n_cpy = multiprocessing.cpu_count()
        
        with tf.device('/cpu:0'):
            self._filenames_phr = tf.placeholder(tf.string, [None], name='filenames')

            dataset = tf.data.Dataset().from_tensor_slices((self._filenames_phr))
            dataset = dataset.map(self._load_data, n_cpy)
            dataset = dataset.flat_map(lambda *samples: tf.data.Dataset().from_tensor_slices(samples))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self._hparams.shuffle_buffer_size, seed=self._hparams.random_seed))
            dataset = dataset.batch(hparams.batch_size)
            dataset = dataset.prefetch(100)

        self._train_iterator = dataset.make_initializable_iterator()
        self._test_iterator = dataset.make_initializable_iterator()

        train_batch = self._train_iterator.get_next()
        test_batch = self._test_iterator.get_next()

        self.train_features = train_batch[0]
        self.train_labels = train_batch[1]

        self.test_features = test_batch[0]
        self.test_labels = test_batch[1]
        
        
        
    def _py_load_data(self, filename):
        data = np.load(os.path.join(self._data_folder, filename.decode()))
        features = data['features']
        labels = data['labels']
        return features, labels

    def _load_data(self, filename):
        features, labels = tf.py_func(self._py_load_data, [filename], [tf.float32, tf.float32])
        features.set_shape([None, self._hparams.n_cand, self._hparams.n_harm])
        labels.set_shape([None])
        return features, labels
    
    def initialize(self, sess):
        sess.run(self._train_iterator.initializer, feed_dict={self._filenames_phr: self._train_filenames})
        sess.run(self._test_iterator.initializer, feed_dict={self._filenames_phr: self._test_filenames})
        