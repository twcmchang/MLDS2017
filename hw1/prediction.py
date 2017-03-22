#-------------------------------------------------------------------------------
# Version:     Python 34
# Purpose:     RNNLM - predict
#
# Author:      cmchang
#
# Created:     March 11, 2017
# Copyright:   (c) cmchang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------


# coding: utf-8

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

# get the index of the maximum argument
def GetMaxArg(arr):
    return max(range(len(arr)), key = lambda x: arr[x])

#  generate the one-hot embedding matrix
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# generate our test batch from testing_data.csv
def TestBatch(myDict,test_path,window=4,cloze='_____'):
    import pandas as pd
    d = pd.read_csv(test_path,sep=",")
    # remove useless characters 
    no_chars = ['(', ')', '[', ']' , ',', '.','!','?','*','"']
    count = 0
    batch = []
    label = []
    for i in range(len(test)):
        ques = []
        ans = []
        sentence = d['question'][i]
        opt = np.asarray(test.filter(regex='\\)').iloc[i])
        for item in no_chars:
            sentence = sentence.replace(item, '')
        sentence = sentence.lower()
        words = sentence.split(' ')    
        if cloze in words:
            pos = words.index(cloze)
            if pos < window:
                for _ in range(window-pos):
                    ques.append(myDict.GetWordIndex('not-a-word'))
                for j in range(pos):
                    ques.append(myDict.GetWordIndex(words[j]))
            else:
                for j in range(pos-window,pos):
                    ques.append(myDict.GetWordIndex(words[j]))
        for j in range(len(opt)):
            ans.append(int(myDict.GetWordIndex(opt[j])))
        batch.append(ques)
        label.append(ans)
    return batch,label

# If there is no existing dictionary and selected sentences, initial = True
# else, initial = False
initial = False

data_path = "/Users/chunmingchang/MLDS2017/hw1/Data"
test_path = "test/testing_data.csv"
ckpt_meta = "/Users/chunmingchang/MLDS2017/hw1/Data/model/model0.314453.ckpt.meta"
ckpt_file = "/Users/chunmingchang/MLDS2017/hw1/Data/model/model0.314453.ckpt"
output_file = "pred_0320"

print("Initial:",str(initial))

if initial:
    # build dictionary
    print('building dictionary ...')
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
    print('selecting training sentences ...')
    SelectTrainSentences(myDict,data_path,0.8,10)

    # revised the following filename 
    selected_sentences = pickle.load(open(os.path.join(data_path,'pickle_sentence_0.80_10_00862361'),'rb'))

else:
    # load in binary by pickle
    print('loading dictionary ...')
    myDict = pickle.load(open('Dict_v0311','rb'))
    myDict.GetWordIndex('test')
    print('loading training sentences ...')
    selected_sentences = pickle.load(open(os.path.join(data_path,'pickle_sentence_0.80_10_00862361'),'rb'))

print("Completed dictionary and selected_sentences ...")

# our embedding matrix, maybe can be replaced by the word embedding of word2vec
embedding_matrix = dense_to_one_hot(np.array(range(len(myDict.both_dict))),num_classes = len(myDict.both_dict))

# hyper parameters
batch_size     = 256
window         = 10 # how many forewords to determine the next words
learning_rate  = 0.1
training_iters = 1000000 # upper bound of epoch * batch_size
display_epoch  = 50 # display per display_epoch 

# Network Parameters
n_input   = len(myDict.both_dict) # because of one-hot encoding here
n_classes = len(myDict.both_dict) # because of one-hot encoding here
n_steps   = window # how many steps in RNN == window
n_hidden  = 256 # hidden layer

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
    biases = tf.Variable(tf.random_normal(bias_shape)) + 0.1

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weight_shape, bias_shape)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph(ckpt_meta)
    saver.restore(sess, ckpt_file)
    
    print("Model restored.")
    
    # Now you can run the model to get predictions
    batch_x, batch_y = TestBatch(myDict,
                                 os.path.join(data_path,test_path),
                                 window = window,
                                 cloze = '_____')
    
    batch_x = embedding_matrix[batch_x]
    P = sess.run(pred,feed_dict={x:batch_x})
    
    final = {}
    option = ['a','b','c','d','e']
    for i in range(0,len(P)):
        # final[i+1] to fit the index which starts from 1 in submission
        final[i+1] = option[GetMaxArg(P[i][batch_y[i]])]
    fw = pd.DataFrame.from_dict(final,orient='index')
    fw.to_csv(output_file+".csv",index_label='id',header=['answer'])

