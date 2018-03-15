import os
import numpy as np
import tensorflow as tf

def get_npz_files(path):
    filenames = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.npz'):
            filenames.append(os.path.join(path, f))
            
    return filenames

def load_file(path):
    data = np.load(path.decode("utf-8"))
    return data['features'], data['labels'][..., np.newaxis]


class Dataset:
    def __init__(self, sess, dataset_path='D:/musan/preprocessed', valid_split=0.15, batch_size=32):     
        dataset_files = np.array(get_npz_files(dataset_path))
        valid_files_quantity = int(len(dataset_files) * valid_split)
        files_indices = np.arange(len(dataset_files))
        np.random.shuffle(files_indices)

        valid_indices = files_indices[:valid_files_quantity]
        train_indices = files_indices[valid_files_quantity:]

        valid_files = dataset_files[valid_indices]
        train_files = dataset_files[train_indices]
        
        files_phr = tf.placeholder(train_files.dtype, name='files_phr')

        dataset = tf.data.Dataset().from_tensor_slices((files_phr))
        dataset = dataset.map(lambda path: tf.py_func(load_file, [path], [tf.float32, tf.float32]))
        dataset = dataset.shuffle(buffer_size=20)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset().from_tensor_slices((x, y)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        
        train_iterator = dataset.make_initializable_iterator()
        valid_iterator = dataset.make_initializable_iterator()
        
        sess.run(train_iterator.initializer, feed_dict={files_phr: train_files})
        sess.run(valid_iterator.initializer, feed_dict={files_phr: valid_files})
        
        self.train_batch = train_iterator.get_next()
        self.valid_batch = valid_iterator.get_next()
        
    def get_train_batch(self):
        return self.train_batch
    
    def get_valid_batch(self):
        return self.valid_batch
    