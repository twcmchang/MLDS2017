# -*- coding: utf-8 -*-
from __future__ import print_function
#import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
#import time
import os
import re
from six.moves import cPickle

#from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=50,
                       help='number of times to sample')
    parser.add_argument('--test_file', type=str, default='data/testing_data.csv',
                       help='test file')
    parser.add_argument('--result_file', type=str, default='result.csv',
                       help='result file')
    parser.add_argument('--sample', type=int, default=0,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            test_data = pd.read_csv(args.test_file)
            OPTIONS = ['a', 'b', 'c', 'd', 'e']
            with open(args.result_file, 'wb') as f_out:
                f_out.write('id,answer\n')
                for n in range(0, test_data.shape[0]):
                    # test_data.shape[0]
                    l = test_data.iloc[n]
                    prime = clean_str(l[1].split(' _____ ')[0])
                    options = [clean_str(opt) for opt in l[2:]]
                    print(str(l[0]))#  + '\n' + l[1].split(' _____ ')[0] + '\n' + prime + '\n' + ','.join(options))
                    f_out.write(str(l[0]) + ',' + OPTIONS[model.sample(sess, words, vocab, args.n, prime, options, args.sample)] + '\n')

if __name__ == '__main__':
    main()
