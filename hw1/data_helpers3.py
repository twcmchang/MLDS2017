import numpy as np
import re
import itertools
import pickle
import os
from collections import Counter
from Dict import Dict

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""
'''
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
'''

def data():
    # If there is no existing dictionary and selected sentences, initial = True
    # else, initial = False
    initial = False

    data_path = "./Data"

    print("Initial:",str(initial))

    if initial:
        # build dictionary
        myDict = Dict(train_path= os.path.join('train_sentence'),
                    test_path = os.path.join(data_path,'testing_data.csv'),
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

    return selected_sentences


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    sequence_length = 50
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence)<=sequence_length:                
            #num_padding = sequence_length - len(sentence)
            #new_sentence = [padding_word] * num_padding + sentence
            padded_sentences.append(sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()] + [' ']
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])


    return [x]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences = data()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    #print (vocabulary['test'], vocabulary_inv[20])
    x = build_input_data(sentences_padded, vocabulary)
    return [x, vocabulary, vocabulary_inv]

'''
a=data()
print a[0]


a,b, c, d=load_data()
print (a[0:10],'\n',c[0:10],d)
'''
