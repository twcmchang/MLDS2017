import os
import numpy as np
import skimage
import skipthoughts
from util import parse_raw_tag_dict, get_tag_dict
import cPickle as pickle

## Remember to download model and table from
## https://github.com/ryankiros/skip-thoughts
## and specify their locations in skipthoughts.py
model = skipthoughts.load_model()

## Download tags_clean.csv from link in HW3 slides
raw_tag_dict = parse_raw_tag_dict('tags_clean.csv')
tag_dict_in_use = get_tag_dict(raw_tag_dict)

## Take tags containing either 'hair' or 'eye'
## (otherwise, it will be: ' and ')
## There are 18175 text left
tag_dict_in_use_1 = dict([(k,v) for k,v in tag_dict_in_use.items() if not v == ' and '])

## Encode texts into 4800-dim vectors
vecs = skipthoughts.encode(model, tag_dict_in_use_1.values(), verbose = False)
new_dict = {}
idx = 0
for k in tag_dict_in_use_1:
    new_dict[k] = vecs[idx]
    idx = idx + 1

## Save these 18175 vectors
pickle.dump(new_dict, open('vec_hair_eyes.p', 'wb'))

## Save a special vector of 'blonde hair blue eyes' for testing
vec_tmp = skipthoughts.encode(model, ['blonde hair blue eyes'], verbose = False)
pickle.dump(vec_tmp, open('blonde_hair_blue_eyes.p', 'wb'))
