#-------------------------------------------------------------------------------
# Version:      Python 34
# Purpose:      RNNLM - to generate batch from selected sentences
#
# Author:       cmchang
#
# Created:      March 10, 2017
# Copyright:    (c) cmchang 2017
# Licence:      <your licence>
#-------------------------------------------------------------------------------
import collections
import random
import numpy as np
from Dict import Dict


def GenerateBatch(myDict,selected_sentences,batch_size,window):
    # if len(selected_sentences):
    #     print('at least one sentence in selected_sentences')
    #     return
    batch = []
    label = []
    buffer = collections.deque(maxlen=window+1)
    while len(batch) < batch_size:
        data_index = 0 
        data = selected_sentences[random.randint(0,len(selected_sentences)-1)]
        for _ in range(window+1):
            buffer.append(myDict.GetWordIndex(data[data_index]))
            data_index = (data_index + 1)
        while data_index < len(data):
            # get a sample
            sample = np.asarray(buffer)
            batch.append(sample[:window])
            label.append(sample[window])
            if len(batch) >= batch_size:
                break
            # update buffer and data_index
            buffer.append(myDict.GetWordIndex(data[data_index]))
            data_index = (data_index + 1)
    return batch,label