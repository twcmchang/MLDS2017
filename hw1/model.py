import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
#import random
import numpy as np
from collections import Counter
#from beam import BeamSearch

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
            
        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data),
                                  args.seq_length, 1)
                #inputs = tf.nn.embedding_lookup(embedding, self.input_data)
                #inputs = tf.split(inputs, args.seq_length, 1)                
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')        
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        # output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='first all', options='a,b,c', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32)) 
        #print (prime)
        
        for word in prime.split()[:-1]:
            # print (word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word, 0) # if not exists, return 0 ('UNK')
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)
         
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            r = int(np.searchsorted(t, np.random.rand(1)*s))
            return(r)
 
        def most_Common(lst):
            data = Counter(lst)
            return data.most_common(1)[0][0]
            
        #ret = prime
        word = prime.split()[-1]
        #options = options.split(',')
        option_inds = [vocab.get(option, 0) for option in options]
        
        answer = []
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word, 0)
            feed = {self.input_data: x, self.initial_state:state}
            probs = sess.run(self.probs, feed)
            p = probs[0][option_inds]
            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)
            
            answer.append(sample)
            #pred = options[sample]
            #ret += ' ' + pred
            #word = pred

        return(most_Common(answer))


