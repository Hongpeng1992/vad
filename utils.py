import cv2
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa.display
import librosa


eps = tf.constant(1e-8, dtype=tf.float32)

def get_spectogram(cnn_output, spect):
    fig = plt.figure(figsize=(10, 5), dpi=120, facecolor='w', edgecolor='k')
    fig.clear()
    fig.add_subplot(2, 1, 1)
    plt.title('Original spectr')
    librosa.display.specshow(librosa.power_to_db(spect, ref=np.max), y_axis='mel', fmax=4000, x_axis='time', sr=SR, hop_length=128)
    fig.add_subplot(2, 1, 1)
    plt.imshow(cnn_output[0, :, :, 0].T)
    
    fig.set_tight_layout(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent='False')
    buf.seek(0)
    
    fig.clear()
    plt.close(fig)
    
    x = np.frombuffer(buf.getvalue(), dtype='uint8')

    return cv2.imdecode(x, cv2.IMREAD_UNCHANGED)


def plot_confusion_matrix(classes_confusion_matrix, classes_str_to_id, normalize=False):   
    cm = classes_confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm, copy=True)

    np.set_printoptions(precision=2)

    fig = plt.figure(figsize=(classes_confusion_matrix.shape[0]+1, classes_confusion_matrix.shape[1]+1),
                     dpi=120, facecolor='w', edgecolor='k')
    fig.clear()
    
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Blues')

    classes = sorted(classes_str_to_id.items(), key=lambda x: x[1])
    classes, _ = zip(*classes)

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=10,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '0.02f'),
                horizontalalignment="center",
                verticalalignment='center',
                fontsize=14,
                color=('black' if cm[i, j] < 0.5 else 'white'))
        
    fig.set_tight_layout(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent='False')
    buf.seek(0)
    
    fig.clear()
    plt.close(fig)
    
    x = np.frombuffer(buf.getvalue(), dtype='uint8')

    return cv2.imdecode(x, cv2.IMREAD_UNCHANGED)


def cross_entropy(targets, predictions, name='cross_entropy'):
    with tf.name_scope(name):
        cost = - targets * tf.log(predictions + eps) - (1 - targets) * tf.log(1 - predictions + eps)
    return cost

def accuracy(classes, targets):
    return tf.reduce_mean(tf.cast(tf.equal(classes, targets), dtype=tf.float32))