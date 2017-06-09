## --------------------------
## Imports and configurations
## --------------------------
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

import numpy as np
import tqdm
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## ----------------------
## Define Model_DNN class
## ----------------------
class Model_DNN(object):
    def __init__(self, model_name = 'DNN', n_steps = 40000,
                 input_size = 784, output_size = 10,
                 batch_size = 64, hidden_size_1 = 100, hidden_size_2 = 100,
                 act_fn = tf.nn.sigmoid, opt_fn = tf.train.GradientDescentOptimizer,
                 learning_rate = 0.01,
                 use_bn = True, momentum = 0.999, epsilon = 1e-3, is_training = True,
                 save_dir='save_DNN'):
        self.model_name = model_name
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.act_fn = act_fn
        self.opt_fn = opt_fn
        self.learning_rate = learning_rate
        self.use_bn = use_bn
        self.momentum = momentum
        self.epsilon = epsilon
        self.is_training = is_training
        self.save_dir = save_dir
    
    def batch_norm_wrapper(self, inputs, is_training, momentum = 0.999, epsilon = 1e-3):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * momentum + batch_mean * (1 - momentum))
            train_var = tf.assign(pop_var,
                                  pop_var * momentum + batch_var * (1 - momentum))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)
    
    def build_model(self, w1_init, w2_init, w3_init):
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.output_size])
        
        # The first hidden layer
        self.w1 = tf.Variable(w1_init)
        if self.use_bn:
            self.z1 = tf.matmul(self.x,self.w1)
            self.bn1 = self.batch_norm_wrapper(self.z1, self.is_training, self.momentum, self.epsilon)
            self.l1 = self.act_fn(self.bn1)
        else:
            self.b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_1]))
            self.z1 = tf.matmul(self.x,self.w1)+self.b1
            self.l1 = self.act_fn(self.z1)
        
        # The second hidden layer
        self.w2 = tf.Variable(w2_init)
        if self.use_bn:
            self.z2 = tf.matmul(self.l1,self.w2)
            self.bn2 = self.batch_norm_wrapper(self.z2, self.is_training, self.momentum, self.epsilon)
            self.l2 = self.act_fn(self.bn2)
        else:
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_2]))
            self.z2 = tf.matmul(self.l1,self.w2)+self.b2
            self.l2 = self.act_fn(self.z2)

        # The output layer: always use softmax
        self.w3 = tf.Variable(w3_init)
        self.b3 = tf.Variable(tf.constant(0.1, shape=[self.output_size]))
        self.y  = tf.nn.softmax(tf.matmul(self.l2, self.w3)+self.b3)

        # Loss, Optimizer and Predictions
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))

        self.train_step = self.opt_fn(self.learning_rate).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.arg_max(self.y,1),tf.arg_max(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))

        # Gradients of loss w.r.t. weights
        self.w2_grad = tf.gradients(self.cross_entropy, [self.w2])[0]
        
        ## Keep the last 5 checkpoints
        self.saver = tf.train.Saver()
    
    def train(self, sess):
        self.w2_list, self.w2_grad_list, self.accuracy_list, self.z2_list = [], [], [], []
        for idx in tqdm.tqdm(range(self.n_steps)):
            batch = mnist.train.next_batch(self.batch_size)
            _, self.w2_, self.w2_grad_ = sess.run([self.train_step, self.w2, self.w2_grad],
                                                  feed_dict={self.x: batch[0], self.y_: batch[1]})
            self.w2_list.append(self.w2_)
            self.w2_grad_list.append(self.w2_grad_)
            if idx % 50 is 0:
                self.accuracy_, self.z2_ = sess.run([self.accuracy, self.z2],
                                                    feed_dict={self.x: mnist.test.images,
                                                               self.y_: mnist.test.labels})
                self.accuracy_list.append(self.accuracy_)
                self.z2_list.append(np.mean(self.z2_, axis = 0))
            if idx % 500 is 0:
                self.save(sess, self.save_dir, idx)
        self.w2_list = np.array(self.w2_list)
        self.w2_grad_list = np.array(self.w2_grad_list)
        self.accuracy_list = np.array(self.accuracy_list)
        self.z2_list = np.array(self.z2_list)
    
    @property
    def model_dir(self):
        model_dir = "bsize{}_{}_{}_{}".format(self.batch_size,
                                           self.act_fn.__qualname__,
                                           self.opt_fn.__qualname__,
                                           str(self.learning_rate)[2:])
        if self.use_bn:
            model_dir = model_dir + '_BN'
        return model_dir
    
    def save(self, sess, save_dir, step):
        model_name = self.model_name + ".model"
        save_dir = os.path.join(save_dir, self.model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(sess, os.path.join(save_dir, model_name), global_step=step)

## -------------------------------------------------------------------------------
## Generate predetermined random weights so the networks are similarly initialized
## -------------------------------------------------------------------------------
w1_initial = tf.truncated_normal([784,100], stddev=np.sqrt(2 / 784), seed=5566)
w2_initial = tf.truncated_normal([100,100], stddev=np.sqrt(2 / 100), seed=5566)
w3_initial = tf.truncated_normal([100,10], stddev=np.sqrt(2 / 100), seed=5566)

## ----------------------------------------------------
## Specify the directory to save checkpoint and figures
## ----------------------------------------------------
save_dir = 'save_DNN_better_init'

## =============================
## (1) Try different batch sizes
## =============================
bsize_list = [4, 16, 64, 256]

w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}
w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}

for bsize in bsize_list:
    ## run model with BN:
    model = Model_DNN(n_steps = 40000, batch_size = bsize, save_dir = save_dir)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_BN_dict[bsize] = model.w2_list
    w2_grad_BN_dict[bsize] = model.w2_grad_list
    accuracy_BN_dict[bsize] = model.accuracy_list
    z2_BN_dict[bsize] = model.z2_list
    ## run model without BN:
    model = Model_DNN(n_steps = 40000, batch_size = bsize, use_bn = False, save_dir = save_dir)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_dict[bsize] = model.w2_list
    w2_grad_dict[bsize] = model.w2_grad_list
    accuracy_dict[bsize] = model.accuracy_list
    z2_dict[bsize] = model.z2_list

## Plot accuracy
fig, ax = plt.subplots(figsize=(12,8))
color_list = ['b', 'g', 'r', 'k', 'c', 'm']

## (sort by accuracy in decreasing order)
final_accuracy_BN_dict = [accuracy_BN_dict[i][400] for i in bsize_list]
decreasing_order = sorted(range(len(final_accuracy_BN_dict)),
                          key=lambda k: -final_accuracy_BN_dict[k])

## Plot accuracy with BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_BN_dict[bsize_list[idx]])*50,50),
            accuracy_BN_dict[bsize_list[idx]],
            color = color_list[idx], alpha = 0.4, linewidth = 2.5,
            label='bsize=%d, BN'%(bsize_list[idx]))

## Plot accuracy without BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_dict[bsize_list[idx]])*50,50),
            accuracy_dict[bsize_list[idx]],
            color = color_list[idx], alpha = 0.8, linewidth = 1.2,
            linestyle = '--',
            label='bsize=%d'%(bsize_list[idx]))

ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.9,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.savefig(os.path.join(save_dir, 'accuracy_vs_batch_sizes.png'))
plt.show()
## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------

## ======================================
## (2) Try different activation functions
## ======================================
## Define leaky-relu with leak = 0.2 and __qualname = 'lrelu'
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

act_fn_list = [tf.nn.relu, lrelu, tf.tanh, tf.nn.softplus]

w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}
w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}

for afn in act_fn_list:
    key = afn.__qualname__
    ## run model with BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, save_dir = save_dir)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_BN_dict[key] = model.w2_list
    w2_grad_BN_dict[key] = model.w2_grad_list
    accuracy_BN_dict[key] = model.accuracy_list
    z2_BN_dict[key] = model.z2_list
    ## run model without BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, use_bn = False, save_dir = save_dir)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_dict[key] = model.w2_list
    w2_grad_dict[key] = model.w2_grad_list
    accuracy_dict[key] = model.accuracy_list
    z2_dict[key] = model.z2_list

## Plot accuracy
fig, ax = plt.subplots(figsize=(12,8))
color_list = ['b', 'g', 'r', 'k', 'c', 'm']

## (sort by accuracy in decreasing order)
final_accuracy_BN_dict = [accuracy_BN_dict[i][400] for i in key_list]
decreasing_order = sorted(range(len(final_accuracy_BN_dict)),
                          key=lambda k: -final_accuracy_BN_dict[k])

## Plot accuracy with BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_BN_dict[key_list[idx]])*50,50),
            accuracy_BN_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.4, linewidth = 2.5,
            label='activation function = %s, BN'%(key_list[idx]))

## Plot accuracy without BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_dict[key_list[idx]])*50,50),
            accuracy_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.8, linewidth = 1.2,
            linestyle = '--',
            label='activation function = %s'%(key_list[idx]))

ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
# ax.set_xlim([20000,40000])
ax.set_ylim([0.92,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.savefig(os.path.join(save_dir, 'accuracy_vs_act_fns.png'))
plt.show()
## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------

## =========================================================
## (3) Try different activation functions with AdamOptimizer
## =========================================================
act_fn_list = [tf.nn.sigmoid, tf.nn.relu, lrelu, tf.tanh, tf.nn.softplus]

w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}
w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}

for afn in act_fn_list:
    key = afn.__qualname__
    ## run model with BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, save_dir = save_dir,
                      opt_fn = tf.train.AdamOptimizer)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_BN_dict[key] = model.w2_list
    w2_grad_BN_dict[key] = model.w2_grad_list
    accuracy_BN_dict[key] = model.accuracy_list
    z2_BN_dict[key] = model.z2_list
    ## run model without BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, use_bn = False, save_dir = save_dir,
                      opt_fn = tf.train.AdamOptimizer)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_dict[key] = model.w2_list
    w2_grad_dict[key] = model.w2_grad_list
    accuracy_dict[key] = model.accuracy_list
    z2_dict[key] = model.z2_list

## Plot accuracy
fig, ax = plt.subplots(figsize=(12,8))
color_list = ['b', 'g', 'r', 'k', 'c', 'm']

## (sort by accuracy in decreasing order)
final_accuracy_BN_dict = [accuracy_BN_dict[i][400] for i in key_list]
decreasing_order = sorted(range(len(final_accuracy_BN_dict)),
                          key=lambda k: -final_accuracy_BN_dict[k])

## Plot accuracy with BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_BN_dict[key_list[idx]])*50,50),
            accuracy_BN_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.4, linewidth = 2.5,
            label='activation function = %s, BN'%(key_list[idx]))

## Plot accuracy without BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_dict[key_list[idx]])*50,50),
            accuracy_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.8, linewidth = 1.2,
            linestyle = '--',
            label='activation function = %s'%(key_list[idx]))

ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
# ax.set_xlim([20000,40000])
ax.set_ylim([0.9,1])
ax.set_title('Batch Normalization Accuracy with AdamOptimizer')
ax.legend(loc=4)
plt.savefig(os.path.join(save_dir, 'accuracy_vs_act_fns_adam.png'))
plt.show()
## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------