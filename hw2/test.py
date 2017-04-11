import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import json
from model import Video_Caption_Generator
from utils import data_preprocess, get_video_feat, get_padding_caption

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_video_feat_path', type=str, default='MLDS_hw2_data/training_data/feat/',
						help='directory which contains training video feature files')
	parser.add_argument('--test_video_feat_path', type=str, default='MLDS_hw2_data/testing_data/feat/',
						help='directory which contains testing video feature files')
	parser.add_argument('--train_label_json', type=str, default='MLDS_hw2_data/training_label.json',
						help='json file of training captions and corresponding video id')
	parser.add_argument('--test_label_json', type=str, default='MLDS_hw2_data/testing_public_label.json',
						help='json file of testing captions and corresponding video id'))
	parser.add_argument('--result_file', type=str, default='output.json',
						help='result file')
	parser.add_argument('--init_from', type=str, default='save',
						help="""initialize from saved model at this path.
						'config.pkl'	: configuration;
						'checkpoint'	: paths to model file(s) (created by tf).
										Note: this file contains absolute paths, be careful when moving files around;
						'model.ckpt-*'	: file(s) with model definition (created by tf)
						"""))

    args = parser.parse_args()
    test(args)

def test(args):

	assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
	# open old config and check if models are compatible
	with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
		saved_args = cPickle.load(f)

	ckpt = tf.train.get_checkpoint_state(args.init_from)

	vocab, vocab_inv, train_feat_id, train_caption, test_feat_id, test_caption = data_preprocess(args.train_label_json,args.test_label_json)

	model = Video_Caption_Generator(args,n_vocab=len(vocab),infer=True)
	
	with tf.Session() as sess:

		tf.global_variables_initializer().run()
		print("Initialized")
		
		saver = tf.train.Saver(tf.global_variables())
		if args.init_from is not None:
			saver.restore(sess, ckpt.model_checkpoint_path)

		result = []

		for i in range(len(test_feat_id)):
			this_test_feat_id = test_feat_id[i]
				
			# get vdieo features
			current_feat, current_feat_mask = get_video_feat(args.test_video_feat_path, this_test_feat_id)

			# randomly select one captions for one video and get padding captions with maxlen = 20
			# current_caption, current_caption_mask = get_padding_caption(vocab, batch_caption, maxlen= model.n_caption_step+1)

			this_gen_idx = sess.run([model.gen_caption_idx],feed_dict={
										model.video: current_feat,
										model.video_mask : current_feat_mask,
										})
			this_gen_words = vocab_inv(this_gen_idx)
			punctuation = np.argmax(np.array(this_gen_words) == '<eos>') + 1
			this_gen_words = this_gen_words[:punctuation]

			this_caption = ' '.join(this_gen_words)
			this_caption = this_caption.replace('<bos> ', '')
			this_caption = this_caption.replace(' <eos>', '')

			this_answer = {}
			this_answer['caption'] = this_caption
			this_answer['id'] = this_test_feat_id

			print('Id: %s, caption: %s' % (this_test_feat_id, this_caption))

			result.append(this_answer)

		with open(args.result_file, 'w') as fout:
    		json.dump(result, fout)

if __name__ == '__main__':
    main()


