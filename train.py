import tensorflow as tf
import struct
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tflearn
from Loss_ASoftmax import Loss_ASoftmax
def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        if kind=='train':
                labels_path=os.path.abspath('/home/ubuntu/mnist/train-labels-idx1-ubyte')
                images_path=os.path.abspath('/home/ubuntu/mnist/train-images-idx3-ubyte')
        else:
                labels_path=os.path.abspath('/home/ubuntu/mnist/t10k-labels-idx1-ubyte')
                images_path=os.path.abspath('/home/ubuntu/mnist/t10k-images-idx3-ubyte')

        with open(labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II',
                                                                 lbpath.read(8))
                labels = np.fromfile(lbpath,
                                                         dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack(">IIII",
                                                                                           imgpath.read(16))
                images = np.fromfile(imgpath,
                                                         dtype=np.uint8).reshape(len(labels), 784)

        return images, labels
def Network(data_input, training = True):
    x = tflearn.conv_2d(data_input, 32, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 32, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.flatten(x)
    feat = tflearn.fully_connected(x, 2, weights_init = 'xavier')
    return feat

class Module(object):

    def __init__(self, batch_size, num_classes):
        x = tf.placeholder(tf.float32, [batch_size, 784])
        y_ = tf.placeholder(tf.int64, [batch_size,])
        I = tf.reshape(x, [-1, 28, 28, 1])
        feat = Network(I)
        dim = feat.get_shape()[-1]
        logits, loss = Loss_ASoftmax(x = feat, y = y_, l = 1.0, num_cls = num_classes, m = 2)
        self.x_ = x
        self.y_ = y_
        self.y = tf.argmax(logits, 1)
        self.feat = feat
        self.loss = loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_), 'float'))

batch_size = 256
BATCH_SIZE=256
num_iters = 2000
num_classes = 10
X_train, y_train = load_mnist('../../mnist', kind='train')
mms=MinMaxScaler()
X_train=mms.fit_transform(X_train)
print X_train.shape
#X_train=np.reshape(X_train,[60000,28,28,1])

batch_len =int( X_train.shape[0]/BATCH_SIZE)
batch_idx=0
train_idx=np.random.permutation(batch_len)



sess = tf.InteractiveSession()
mod = Module(batch_size, num_classes)
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 10000, 0.9, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(mod.loss, global_step)

tf.global_variables_initializer().run()

for t in range(num_iters):
        batch_shuffle_idx=train_idx[batch_idx]
        batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
        batch_ys=y_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]

        if batch_idx<batch_len:
                batch_idx+=1
                if batch_idx==batch_len:
                        batch_idx=0
        else:
                batch_idx=0

        fd = { mod.x_ : batch_xs, mod.y_ : batch_ys }
        _, v = sess.run([train_op, mod.loss], feed_dict=fd)
        if t % 10 == 0:
             print (t, v)
	     acc=sess.run(mod.accuracy, feed_dict={mod.x_: X_train[:256], mod.y_:y_train[:256]})
	     print acc
print ('Training Done')

