#-------------------------------------------------------------------------------
# Version:     Python 34
# Purpose:     RNNLM - the main script to train a RNNLM
#
# Author:      cmchang
#
# Created:     March 11, 2017
# Copyright:   (c) cmchang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from Dict import Dict
from generate_batch import GenerateBatch
from select_sentence import SelectTrainSentences
import os
import pickle
import random
import collections
import pandas as pd
import numpy as np
import tensorflow as tf

# If there is no existing dictionary and selected sentences, initial = True
# else, initial = False
initial = False

data_path = "/Users/chunmingchang/MLDS2017/hw1/Data"

print("Initial:",str(initial))

if initial:
    # build dictionary
    myDict = Dict(train_path= os.path.join(data_path,'train_sentence'),
                test_path = os.path.join(data_path,'test/testing_data.csv'),
                lb=1000,ub=100000)
    # testing
    myDict.GetWordIndex('test')

    # save dictionary in binary by pickle 
    output = open('Dict_v0311', 'wb')
    pickle.dump(myDict, output)
    output.close()

    # select training sentences with c_threshold=0.8 and word_limit=10
    # save selected sentences under data_path
    
    SelectTrainSentences(myDict,data_path,0.8,10)

    # revised the following filename 
    selected_sentences = pickle.load(open(os.path.join(data_path,'pickle_sentence_0.80_10_00862361'),'rb'))

else:
    # load in binary by pickle
    myDict = pickle.load(open('Dict_v0311','rb'))
    myDict.GetWordIndex('test')
    selected_sentences = pickle.load(open(os.path.join(data_path,'pickle_sentence_0.80_10_00862361'),'rb'))

print("Completed dictionary and selected_sentences ...")

# function: generate one hot embedding matrix
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# our embedding matrix, maybe can be replaced by the word embedding of word2vec
embedding_matrix = dense_to_one_hot(np.array(range(len(myDict.both_dict))),num_classes = len(myDict.both_dict))


# hyper parameters
batch_size     = 128
window         = 4 # how many forewords to determine the next words
learning_rate  = 0.001
training_iters = 5000000 # upper bound of epoch * batch_size
display_epoch  = 50 # display per display_epoch 

# Network Parameters
n_input   = len(myDict.both_dict) # because of one-hot encoding here
n_classes = len(myDict.both_dict) # because of one-hot encoding here
n_steps   = window # how many steps in RNN == window
n_hidden  = 128 # hidden layer

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# weights shape
weight_shape = [n_hidden,n_classes]
# bias shape
bias_shape = [n_classes]

# In[10]:

def RNN(x, weight_shape, bias_shape):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    
    # Define weights
    weights = tf.Variable(tf.random_normal(weight_shape))
    biases = tf.Variable(tf.random_normal(bias_shape))
    
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases


# In[11]:

pred = RNN(x, weight_shape, bias_shape)


# In[12]:

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[13]:

# Initializing the variables
init = tf.global_variables_initializer()

#
saver = tf.train.Saver()

print("Starting training RNN ...")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    bestaccu = 0.0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # generate a batch of samples
        batch_x, batch_y = GenerateBatch(myDict,selected_sentences,batch_size,window)

        # convert to one hot encoding
        batch_x = embedding_matrix[batch_x]
        batch_y = embedding_matrix[batch_y]
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

        if step % display_epoch == 0:
            print(  "Iter " + str(step*batch_size) +
                    ", Minibatch Loss= "    + "{:.6f}".format(loss) +
                    ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
        if acc > bestaccu:
            bestaccu = acc
	    # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(data_path,"model"+str(acc)+".ckpt"))
            print("Model saved in file: %s" % save_path)
    print("Optimization Finished!")

#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
