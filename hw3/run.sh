#!/bin/bash
wget -O model.zip "https://www.dropbox.com/s/xplxts5g9ousjs6/model.zip?dl=0"
unzip model.zip -d ./
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget -P sent2vec/ http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
python3 generate.py --testing_text "$1" --sample_dir samples --init_from model/
