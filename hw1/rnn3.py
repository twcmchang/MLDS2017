import collections
import numpy as np
import tensorflow as tf
import data_helpers3


#-------------------------------数据预处理---------------------------#

x, vocabulary, vocabulary_inv = data_helpers3.load_data()
x_data=x[0]
#print (x_data[:5])
nb_word = len(vocabulary_inv)

batch_size = 64
n_chunk = len(x_data) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size

	batches = x_data[start_index:end_index]
	length = max(map(len,batches))
	xdata = np.full((batch_size,length), vocabulary[' '], np.int32)
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata)
	y_batches.append(ydata)
#print (x_batches[:5])

#---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.contrib.rnn.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.contrib.rnn.GRUCell
	elif model == 'lstm':
		cell_fun = tf.contrib.rnn.BasicLSTMCell

	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

	initial_state = cell.zero_state(batch_size, tf.float32)

	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, nb_word+1])
		softmax_b = tf.get_variable("softmax_b", [nb_word+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [nb_word+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)

	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])

	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
#训练
def train_neural_network():
	logits, last_state, _, _, _ = neural_network()
	targets = tf.reshape(output_targets, [-1])
	loss = tf.contrib.legacy_seq2seq.sequence_loss([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], nb_word)
	cost = tf.reduce_mean(loss)
	learning_rate = tf.Variable(0.0, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	with tf.Session() as sess:
		# Initializing the variables
		#init = tf.global_variables_initializer()
		sess.run(tf.global_variables_initializer())
		#
		saver = tf.train.Saver()

		for epoch in range(1):
			sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			n = 0
			for batche in range(n_chunk):
				train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
				n += 1
				print(epoch, batche, train_loss)
			#if epoch % 7 == 0:
				#saver.save(sess, 'poetry.module', global_step=epoch)

train_neural_network()
