import tensorflow as tf
import numpy as np

class Video_Caption_Generator():
	def __init__(self, dim_image, n_video_step, n_caption_step, dim_hidden, n_lstm_step, batch_size, n_vocab, learning_rate, grad_clip):
		# model parameters
		self.dim_image 		= dim_image
		self.n_video_step 	= n_video_step
		self.n_caption_step = n_caption_step
		self.dim_hidden 	= dim_hidden
		self.n_lstm_step 	= n_lstm_step
		self.batch_size 	= batch_size
		self.n_vocab 		= n_vocab
		self.learning_rate	= learning_rate
		self.grad_clip		= grad_clip

		# model components
		# two LSTM layers
		# word embedding for input
		with tf.device("/cpu:0"):
			self.Wemb = tf.Variable(tf.random_uniform([n_vocab, dim_hidden],-0.1,0.1), name="Wemb")
		# with tf.device("/cpu:0"):
		# 	self.Wemb = tf.get_variable("Wemb",tf.random_uniform([self.n_vocab, self.dim_hidden],-0.1,0.1))
		
		#with tf.variable_scope("model"):
		#with tf.variable_scope("LSTM1"):
		self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
		#with tf.variable_scope("LSTM2"):
		self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

		# image embedding
		self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden],-0.1,0.1), name="embed_image_W")
		self.embed_image_b = tf.Variable(tf.zeros([dim_hidden]), name="embed_image_b")
		# self.embed_image_W = tf.get_variable("embed_image_W",tf.random_uniform([dim_image, dim_hidden],-0.1,0.1))
		# self.embed_image_b = tf.get_variable("embed_image_b",tf.zeros([dim_hidden]))


		# word embedding for output
		self.embed_word_W  = tf.Variable(tf.random_uniform([dim_hidden,n_vocab],-0.1,0.1), name="embed_word_W")
		self.embed_word_b  = tf.Variable(tf.zeros([n_vocab]), name="embed_word_b")
		# self.embed_word_W  = tf.get_variable("embed_word_W",tf.random_uniform([dim_hidden,n_vocab],-0.1,0.1))
		# self.embed_word_b  = tf.get_variable("embed_word_b",tf.zeros([n_vocab]))

		# def build_model(self):
		# 	# input
		self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_step, self.dim_image])
		self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_step])

		# output
		self.caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_step+1]) # int32 for embedding_look_up
		self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_step+1])

		# results
		pred_probs = []
		loss = 0.0

		### Step1: transform video into embedded video ###
		video_flat = tf.reshape(self.video, [-1, self.dim_image]) # (batch_size*n_video_step, dim_image)
		image_embed = tf.nn.xw_plus_b(video_flat, self.embed_image_W, self.embed_image_b)
		image_embed = tf.reshape(image_embed, [self.batch_size, self.n_video_step, self.dim_hidden])

		### Step2: (encoding stage) embedded images into LSTM ###
		# initialization states in 2 LSTM layers
		state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
		state2 = tf.zeros([self.batch_size, self.lstm2.state_size])

		# padding when entering LSTM2
		# A word is embedded as a vector of dim_hidden by Wemb, so the padding vector is with dim_hidden
		padding = tf.zeros([self.batch_size, self.dim_hidden]) 

		# with tf.variable_scope("model",reuse=True):
		for i in range(0,self.n_video_step):
		# if i>0:
		# 	tf.get_variable_scope().reuse_variables()
		# with tf.variable_scope(name_or_scope="LSTM1",reuse=True):
			if i == 0:
				with tf.variable_scope("LSTM1"):
					output1, state1 = self.lstm1(image_embed[:,i,:],state1)
				with tf.variable_scope("LSTM2"):
					output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
			else:
				with tf.variable_scope("LSTM1",reuse=True):
					output1, state1 = self.lstm1(image_embed[:,i,:],state1)
				with tf.variable_scope("LSTM2",reuse=True):
					output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)

		### Step 3: (decoding stage) acquire prediction from LSTM2 output ###

		for i in range(0, self.n_caption_step):
			#with tf.variable_scope('model'):
			#tf.get_variable_scope().reuse_variables()
			with tf.device("/cpu:0"):
				current_word_embed = tf.nn.embedding_lookup(self.Wemb, self.caption[:,i])
			# if i == 0:
			# 	with tf.variable_scope("LSTM1"):
			# 		output1, state1 = self.lstm1(padding, state1)
			# 	with tf.variable_scope("LSTM2"):
			# 		output2, state2 = self.lstm2(tf.concat([current_word_embed,output1],1),state2)
			# else:
			with tf.variable_scope("LSTM1",reuse=True):
				output1, state1 = self.lstm1(padding, state1)
			with tf.variable_scope("LSTM2",reuse=True):
				output2, state2 = self.lstm2(tf.concat([current_word_embed,output1],1),state2)

			### Step 4: calculate loss ### 
			# with tf.variable_scope("model"):
			# 	# create answer 
			answer_index_in_vocab = self.caption[:,i+1]
			answer_index_in_vocab = tf.expand_dims(answer_index_in_vocab, 1)
			batch_index	= tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
			sparse_mat 	= tf.concat([batch_index, answer_index_in_vocab], 1)
			answer_in_onehot = tf.sparse_to_dense(sparse_mat, np.asarray([self.batch_size, self.n_vocab]), 1.0, 0.0)

			# acquire output
			logits = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=answer_in_onehot)
			cross_entropy = cross_entropy * self.caption_mask[:,i]

			# 
			pred_probs.append(logits)

			this_loss = tf.reduce_sum(cross_entropy)/self.batch_size
			loss = loss + this_loss

		self.tf_loss = loss
		self.tf_probs = pred_probs

		# tvars = tf.trainable_variables()
		# grads, _ = tf.clip_by_global_norm(tf.gradients(self.tf_loss, tvars), clip_norm = self.grad_clip)
		# optimizer = tf.train.AdamOptimizer(self.learning_rate)
		# self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.tf_loss)

	# def get_generator():
	# 	return True





