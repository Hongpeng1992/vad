import tensorflow as tf
from conv_rmm_cell import ConvCell

def get_model(features, model, mean, std, is_training, return_activations):
    N_HIDDEN = model['N_HIDDEN']
    N_CAND = model['N_CAND']
    SEQ_LEN = model['SEQ_LEN']
    BATCH_SIZE = model['BATCH_SIZE']
    RNN_STRIDE = model['RNN_STRIDE']
    PADDING = model['PADDING']
    
    with tf.name_scope('standardization'):
        nnet = features - mean
        nnet /= std

    with tf.variable_scope('CNN'):
        nnet = tf.layers.conv2d(nnet, N_HIDDEN, kernel_size=[1, 1], activation=tf.nn.relu, name='Cond2D_1')
        conv_output = tf.layers.conv2d(nnet, 1, kernel_size=[1, 1], activation=tf.nn.sigmoid, name='Cond2D_2')

    with tf.name_scope('CNN_output'):
        conv_act = conv_output[:, -1]
        conv_predictions = tf.reduce_max(conv_act, axis=1)
        
        if is_training:
            conv = tf.reshape(conv_output, [-1, N_CAND])
            conv_predictions_train = tf.reduce_max(conv, axis=1)

    with tf.variable_scope('RNN'):
        cell = ConvCell((SEQ_LEN, N_CAND, 1), filters=1, kernel_size=32, stride=RNN_STRIDE, padding=PADDING, activation=tf.nn.sigmoid, name='recurent_cell')
        rnn_output, state = tf.nn.dynamic_rnn(cell, conv_output, sequence_length=[SEQ_LEN] * BATCH_SIZE, dtype=tf.float32)

    with tf.name_scope('RNN_output'):
        nnet = tf.transpose(rnn_output, (1, 0, 2, 3))
        nnet = tf.gather(nnet, int(nnet.get_shape()[0]) - 1, name="last_rnn_output")
    #     nnet = tf.reshape(rnn_output, [-1, N_CAND])

    logits = tf.reduce_max(nnet, axis=(1, 2))
    # predictions = tf.sigmoid(logits)
    predictions = logits
        
    if is_training:
        predictions = predictions, conv_predictions_train
        
        if return_activations:
            activations = nnet, conv_act
            return predictions, activations
        
        return predictions
    
    predictions = predictions, conv_predictions
    if return_activations:
        activations = nnet, conv_act
        return predictions, activations

    return predictions