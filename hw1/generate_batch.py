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

def generate_batch(selected_sentences,batch_size,window):
    candidate_sent = selected_sentences
    if len(candidate_sent):
        print('at least one sentence in selected_sentences')
        return
    batch = []
    label = []
    buffer = collections.deque(maxlen=window+1)
    while len(batch) < batch_size:
        data_index = 0 
        data = candidate_sent[random.randint(0,len(candidate_sent)-1)]
        for _ in range(window+1):
            buffer.append(data[data_index])
            data_index = (data_index + 1)
        while data_index < len(data):
            # get a sample
            sample = np.asarray(buffer)
            batch.append(sample[:window])
            label.append(sample[window])
            if len(batch) >= batch_size:
                break
            # update buffer and data_index
            buffer.append(data[data_index])
            data_index = (data_index + 1)
    return batch,label