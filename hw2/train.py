import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
from six.moves import cPickle
from model import Video_Caption_Generator
from utils import data_preprocess, get_video_feat, get_padding_caption

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_video_feat_path', type=str, default='MLDS_hw2_data/training_data/feat/',
						help='directory to store checkpointed models')
	parser.add_argument('--test_video_feat_path', type=str, default='MLDS_hw2_data/testing_data/feat/',
						help='directory to store checkpointed models')
	parser.add_argument('--train_label_json', type=str, default='MLDS_hw2_data/training_label.json',
						help='directory to store checkpointed models')
	parser.add_argument('--test_label_json', type=str, default='MLDS_hw2_data/testing_public_label.json',
						help='directory to store checkpointed models')
	parser.add_argument('--save_dir', type=str, default='save',
						help='directory to store checkpointed models')
	parser.add_argument('--dim_image', type=int, default=4096,
						help='dimension of input image')
	parser.add_argument('--dim_hidden', type=int, default=1000,
						help='dimension of LSTM hidden state')
	parser.add_argument('--n_lstm_step', type=int, default=80,
						help='number of LSTM steps')
	parser.add_argument('--n_video_step', type=int, default=80,
						help='number of video steps')
	parser.add_argument('--n_caption_step', type=int, default=20,
						help='number of caption steps')
	parser.add_argument('--n_epoch', type=int, default=1000,
						help='number of epochs')
	parser.add_argument('--batch_size', type=int, default=50,
						help='minibatch size')
	parser.add_argument('--save_every', type=int, default=10,
						help='save frequency')
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help='learning rate')
	parser.add_argument('--grad_clip', type=float, default=10.,
						help='clip gradients at this value')
	parser.add_argument('--gpu_mem', type=float, default=0.800,
						help='% of gpu memory to be allocated to this process. Default is 80%')
	parser.add_argument('--log_dir', type=str, default='log',
						help='directory containing log text')
	parser.add_argument('--schedule_sampling', type=float, default=0.0,
						help='probability of sampling word from prediction')
	parser.add_argument('--attention', type=int, default=0,
						help='open(1) or close(0) the attention mechanism')
	parser.add_argument('--init_from', type=str, default=None,
						help="""continue training from saved model at this path. Path must contain files saved by previous training process:
						    'config.pkl'        : configuration;
						    'vocab.pkl'			: vocabuary;
						    'checkpoint'        : paths to model file(s) (created by tf).
						                          Note: this file contains absolute paths, be careful when moving files around;
						    'model.ckpt-*'      : file(s) with model definition (created by tf)
						""")

	args = parser.parse_args()
	train(args)

def train(args):
	if args.init_from is not None:
		# check if all necessary files exist
		assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
		
		# get ckpt
		ckpt = tf.train.get_checkpoint_state(args.init_from)

		# get vocab
		with open(os.path.join(args.init_from, 'vocab.pkl'), 'rb') as f:
			vocab = cPickle.load(f)
		vocab_inv = {v:k for k, v in vocab.items()}

		# read data
		_, _, train_feat_id, train_caption, test_feat_id, test_caption = data_preprocess(args.train_label_json,args.test_label_json)
		
		# open old config and check if models are compatible
		with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
			saved_args = cPickle.load(f)
		need_be_same=["dim_image","dim_hidden","n_lstm_step","n_video_step","n_caption_step"]
		for checkme in need_be_same:
			assert vars(saved_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
		
		# complete arguments to fulfill different versions
		if("attention" in vars(saved_args)):
			print("attention: %d" % vars(saved_args)["attention"])
		else:
			vars(saved_args)["attention"] = 0

		if("schedule_sampling" in vars(saved_args)):
			print("schedule_sampling: %d" % vars(saved_args)["schedule_sampling"])
		else:
			vars(saved_args)["schedule_sampling"] = 0.0

	else:
		with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
			cPickle.dump(args, f)

		vocab, vocab_inv, train_feat_id, train_caption, test_feat_id, test_caption = data_preprocess(args.train_label_json,args.test_label_json)

		with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
			cPickle.dump(vocab, f)

	model = Video_Caption_Generator(args,n_vocab=len(vocab),infer=False)
	
	# add gpu options
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		tf.global_variables_initializer().run()
		print("Initialized")
		
		saver = tf.train.Saver(tf.global_variables())
		if args.init_from is not None:
			saver.restore(sess, ckpt.model_checkpoint_path)

		loss_fd = open('log/loss.txt', 'w')
		loss_to_draw = []

		for epoch in range(0, args.n_epoch):
			if (model.schedule_sampling > 0.0):
				# [pseudo] prob of schedule sampling linearly increases with epochs
				model.schedule_sampling = np.min([model.schedule_sampling * (1.0+epoch/50),1.0])

			# shuffle 
			index = np.array(range(len(train_feat_id)))
			np.random.shuffle(index)
			epoch_train_feat_id	 = train_feat_id[index]
			epoch_train_caption = train_caption[index]

			loss_to_draw_epoch = []

			for start, end in zip(
					range(0, len(epoch_train_feat_id), model.batch_size),
					range(model.batch_size, len(epoch_train_feat_id), model.batch_size)):
			# for start,end in zip(range(0,2,2),range(2,4,2)):
				start_time = time.time()

				# get one minibatch
				batch_feat_id = epoch_train_feat_id[start:end]
				batch_caption = epoch_train_caption[start:end]
				
				# get vdieo features
				current_feat, current_feat_mask = get_video_feat(args.train_video_feat_path, batch_feat_id)

				# randomly select one captions for one video and get padding captions with maxlen = 20
				current_caption, current_caption_mask = get_padding_caption(vocab, batch_caption, maxlen= model.n_caption_step+1)

				# run train_op to optimizer tf_loss
				_, loss_val = sess.run([model.train_op, model.tf_loss],feed_dict={
							model.video: current_feat,
							model.video_mask : current_feat_mask,
							model.caption: current_caption,
							model.caption_mask: current_caption_mask
							})
				loss_to_draw_epoch.append(loss_val)

				print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
				loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
			if np.mod(epoch, args.save_every) == 0:
				checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=epoch)
				print("Epoch ", epoch, "model saved to {}".format(checkpoint_path))
		loss_fd.close()

if __name__ == '__main__':
    main()


