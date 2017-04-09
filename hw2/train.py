import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
from model import Video_Caption_Generator
from utils import data_preprocess, get_video_feat, get_padding_caption

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/Holmes_Training_Data',
	help='data directory containing input.txt')
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
	# parser.add_argument('--log_dir', type=str, default='logs',
	# help='directory containing tensorboard logs')
	# parser.add_argument('--init_from', type=str, default=None,
	# help="""continue training from saved model at this path. Path must contain files saved by previous training process:
	#     'config.pkl'        : configuration;
	#     'words_vocab.pkl'   : vocabulary definitions;
	#     'checkpoint'        : paths to model file(s) (created by tf).
	#                           Note: this file contains absolute paths, be careful when moving files around;
	#     'model.ckpt-*'      : file(s) with model definition (created by tf)
	# """)

	args = parser.parse_args()
	train(args)

def train(args):
	vocab, vocab_inv, train_feat_id, train_caption, test_feat_id, test_caption = data_preprocess(args.train_label_json,args.test_label_json)

	model = Video_Caption_Generator(
			dim_image = args.dim_image,
			dim_hidden = args.dim_hidden,
			batch_size = args.batch_size,
			n_video_step = args.n_video_step,
			n_caption_step = args.n_caption_step,
			n_lstm_step	= args.n_lstm_step,
			n_vocab = len(vocab),
			learning_rate = args.learning_rate,
			grad_clip = args.grad_clip)

	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		init.run()
		print("Initialized")
		# train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
		saver = tf.train.Saver(max_to_keep=100, write_version=1)
		tf.global_variables_initializer().run()

		loss_fd = open('loss.txt', 'w')
		loss_to_draw = []

		for epoch in range(0, args.n_epoch):
			loss_to_draw_epoch = []

			index = np.array(range(len(train_feat_id)))
			np.random.shuffle(index)
			epoch_train_feat_id	 = train_feat_id[index]
			epoch_train_caption = train_caption[index]

			# current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
			# current_train_data = current_train_data.reset_index(drop=True)

			for start, end in zip(
					range(0, len(epoch_train_feat_id), model.batch_size),
					range(model.batch_size, len(epoch_train_feat_id), model.batch_size)):

				start_time = time.time()

				batch_video_id = epoch_train_feat_id[start:end]
				batch_caption = epoch_train_caption[start:end]
				
				# get vdieo features
				current_video_feat, current_video_feat_mask = get_video_feat(args.train_video_feat_path, batch_video_id)

				# randomly select one captions for one video and get padding captions with maxlen = 20
				current_caption, current_caption_mask = get_padding_caption(vocab, batch_caption, maxlen= model.n_caption_step+1)

				_, loss_val = sess.run([model.train_op, model.tf_loss],feed_dict={
							model.video: current_video_feat,
							model.video_mask : current_video_feat_mask,
							model.caption: current_caption,
							model.caption_mask: current_caption_mask
							})
				loss_to_draw_epoch.append(loss_val)

				print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
				loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

			# # draw loss curve every epoch
			# loss_to_draw.append(np.mean(loss_to_draw_epoch))
			# plt_save_dir = "./loss_imgs"
			# plt_save_img_name = str(epoch) + '.png'
			# plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
			# plt.grid(True)
			# plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

			if np.mod(epoch, args.save_every) == 0:
				print("Epoch ", epoch, " is done. Saving the model ...")
				saver.save(sess, os.path.join(args.save_dir, 'model'), global_step=epoch)
		loss_fd.close()

# def test():
if __name__ == '__main__':
    main()


