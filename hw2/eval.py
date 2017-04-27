import math
import operator
import sys
import json
import argparse
import numpy as np
from functools import reduce

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_label_json', type=str, default='MLDS_hw2_data/testing_public_label.json',
                        help='json file of testing captions and corresponding video id')
    parser.add_argument('--result_file', type=str, default='output.json',
                        help='result file')
    args = parser.parse_args()
    score, caption_len = eval_json(args.result_file,args.test_label_json)
    print('Average BLEU score: %4f' % np.mean(score))
    print('Average caption length: %4f' % np.mean(caption_len))

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t):
    candidate = [s.strip()]
    references = [[t.strip()]] 
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score

def eval_json(output_json, test_json):
    with open(output_json) as json_data:
        output = json.load(json_data)
    # output
    output_caption = []
    output_feat_id = []
    for i in range(len(output)):
        output_caption.append(output[i]['caption'])
        output_feat_id.append(output[i]['id'])

    with open(test_json) as json_data:
        target = json.load(json_data)
    # target
    target_caption = []
    target_feat_id = []
    for i in range(len(target)):
        target_caption.append(target[i]['caption'])
        target_feat_id.append(target[i]['id'])
    #
    score = []
    caption_len = []
    for i in range(len(output_feat_id)):
        output_i = output_caption[i]
        target_i = target_caption[i]
        this_score = 0.0
        for j in range(len(target_i)):
            this_score = this_score + BLEU(output_i,target_i[j])
        score.append(this_score/len(target_i))
        caption_len.append(len(output_i.strip().split()))
    return score, caption_len

if __name__ == '__main__':
    main()