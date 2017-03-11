# coding: utf-8
import os
import re
import glob
from nltk import sent_tokenize

# get the filelist of training data
dirn = "Data/train"
filelist = glob.glob(os.path.join(dirn,'*.TXT'))

# tokenize sentences by nltk sent_tokenize
sentences = []
for i in range(len(filelist)):
    with open(os.path.join(filelist[i]),'r') as f:
        print("sent_tokenizing:"+str(i),end='\r')
        sentence = sent_tokenize(f.read())
        for sent in sentence:
            sentences.append(re.sub('\n',' ',sent).lower())

# save in "train_sentence"
f = open('train_sentence','w')
f.write("\n".join(sentences))
f.close()

