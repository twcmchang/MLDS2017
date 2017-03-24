#-------------------------------------------------------------------------------
# Version:     Python 34
# Purpose:     RNNLM - to select more informative sentences
#
# Author:      cmchang
#
# Created:     March 10, 2017
# Copyright:   (c) cmchang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from Dict import Dict
import os
import re
import pickle

def SelectTrainSentences(big_dict,data_path,c_threshold, word_limit):
    selected_lines = []
    total_lines  = 0
    selected_cnt = 0
    for idx,items in enumerate(big_dict.train_sentences):
        test_cnt  = 0.0
        train_cnt = 0.0
        item_cnt  = len(items)
        total_lines += 1.0
        for item in items:
            if item in big_dict.test_dict:
                test_cnt  += 1.0
            if item in big_dict.train_dict:
                train_cnt += 1.0
        test_cnt  =  test_cnt/item_cnt
        train_cnt = train_cnt/item_cnt
        if test_cnt > c_threshold and train_cnt > c_threshold and item_cnt >= word_limit:
            selected_lines.append(items)
    selected_cnt = len(selected_lines)
    '''...write selected sentences...'''
    print('c_threshold %.04lf / Selected Sz %8d / %8d' % (c_threshold, selected_cnt, total_lines))
    output = open(os.path.join(data_path,'pickle_sentence_%.02f' % (c_threshold)), 'wb')
    pickle.dump(selected_lines, output)
    output.close()
