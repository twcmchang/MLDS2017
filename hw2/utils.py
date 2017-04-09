import os
import numpy as np
import re
import json
import random

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " is", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"\s{0,},", " ", string)
	string = re.sub(r"!", "", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.lower().strip()

def build_word_vocab(sentences, word_count_threshold=1): 
	# borrowed this function from NeuralTalk
	print('preprocessing and creating vocab based on word count threshold %d' % (word_count_threshold))
	word_counts = {}
	nsents = 0
	for sent in sentences:
		nsents += 1
		sent = clean_str(sent)
		for w in sent.lower().split(' '):
			word_counts[w] = word_counts.get(w, 0) + 1
	del word_counts[""] # removed the empty string

	vocab_tmp = [w for w in word_counts if word_counts[w] >= word_count_threshold]
	print('filtered words from %d to %d' % (len(word_counts), len(vocab_tmp)))

	# Build mapping
	vocab_inv = {}
	vocab_inv[0] = '<BOS>'
	vocab_inv[1] = '<EOS>'
	vocab_inv[2] = '<PAD>'
	vocab_inv[3] = '<UNK>'

	vocab = {}
	vocab['<BOS>'] = 0
	vocab['<EOS>'] = 1
	vocab['<PAD>'] = 2
	vocab['<UNK>'] = 3

	idx = 4
	for w in vocab_tmp:
		vocab[w] = idx
		vocab_inv[idx] = w
		idx += 1

	return vocab, vocab_inv

def data_preprocess(train_file,test_file):
	if ('json' not in train_file) or ('json' not in test_file):
		assert 'input_files should be JSON.'
	with open(train_file) as json_data:
		train = json.load(json_data)
	with open(test_file) as json_data:
		test = json.load(json_data)

	train_caption = []
	train_feat_id = []
	for i in range(len(train)):
		train_caption.append(train[i]['caption'])
		train_feat_id.append(train[i]['id'])
	
	train_caption = np.array(train_caption)
	train_feat_id = np.array(train_feat_id)

	test_caption = []
	test_feat_id = []
	for i in range(len(test)):
		test_caption.append(test[i]['caption'])
		test_feat_id.append(test[i]['id'])

	test_caption = np.array(test_caption)
	test_feat_id = np.array(test_feat_id)

	all_caption = np.append(np.hstack(train_caption),np.hstack(test_caption))

	vocab, vocab_inv = build_word_vocab(all_caption)

	return vocab, vocab_inv, train_feat_id, train_caption, test_feat_id, test_caption

def split_padding_caption(vocab, caption, maxlen):	
	caption = clean_str(caption)
	caption = caption.lower().split(' ')
	caption = np.append(['<BOS>'],caption)
	caption = np.append(caption,['<EOS>'])
	caplen = len(caption)
	if caplen < maxlen:
		caption = np.append(caption, np.repeat('<PAD>',maxlen-caplen))
		mask = np.append(np.ones(caplen), np.zeros(maxlen-caplen))
	else:
		caption = caption[:maxlen]
		mask = np.ones(maxlen)
	
	idx_caption = []	
	for w in caption: # append sentences
		if w not in vocab:
			idx_caption.append(vocab['<UNK>'])
		else:
			idx_caption.append(vocab[w])	
	# to numpy array
	idx_caption = np.asarray(idx_caption)
	return idx_caption, mask

def get_padding_caption(vocab, batch_caption, maxlen):
	"""
	For one video, randomly select one caption from all candidate captions 
	"""
	return_caption = []
	return_caption_mask = []
	for i in range(len(batch_caption)):
		sel = random.randint(0,len(batch_caption[i])-1)
		idx_caption, mask = split_padding_caption(vocab, batch_caption[i][sel], maxlen)
		return_caption.append(idx_caption)
		return_caption_mask.append(mask)
	
	return_caption = np.vstack(return_caption).astype(int)
	return_caption_mask = np.vstack(return_caption_mask)

	return return_caption, return_caption_mask

def get_video_feat(feat_path, batch_video_id):
	feature = []
	feature_mask = []
	for i in range(len(batch_video_id)):
		fn = os.path.join(feat_path, batch_video_id[i]+'.npy')
		if os.path.exists(fn):
			d = np.load(fn)
			feature.append(d)
			feature_mask.append(np.ones(d.shape[0]))
		else:
			print('%s does not exist' % (fn))
	feature = np.asarray(feature)
	feature_mask = np.asarray(feature_mask)
	return feature, feature_mask

