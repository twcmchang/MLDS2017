#-------------------------------------------------------------------------------
# Version:     Python 34
# Purpose:     RNNLM - to build dictionary 
#
# Author:      cmchang
#
# Created:     March 10, 2017
# Copyright:   (c) cmchang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

class Dict(object):
    def __init__(self,train_path,test_path,lb,ub,both_dict=None):
        self.test_dict       = None
        self.train_dict      = None
        self.sort_test_list  = None
        self.sort_train_list = None
        self.train_sentences = None
        self.both_dict       = both_dict
        if both_dict is None:
            print("construct dict ...")
            self.BuildDictFromTest(test_path)
            self.BuildDictFromTrain(train_path,lb,ub)
            self.BothDict()
            self.ConstructIndex()
        self.dic_size = len(self.both_dict)
    
    def GetWordIndex(self, tar_word):
        if tar_word in self.both_dict:
            return self.both_dict[tar_word][0]
        else:
            return self.both_dict[self.not_in_bag][0]
    
    def ConstructIndex(self):
        index_start = 0
        for key in self.both_dict.keys():
            self.both_dict[key][0] = index_start
            index_start += 1

    def BuildDictFromTest(self, test_path):
        print('build dictionary from '+test_path)
        self.test_dict = {}
        
        # read testing_data.csv
        test = pd.read_csv(test_path,sep=",")
        
        # remove useless characters 
        no_chars = ['(', ')', '[', ']' , ',', '.','!','?','*','"']

        # columns of data to build dictionary of testing data
        interest = test.drop('id',1).columns.values
        for intr in interest: # selected
            for line in test[intr]:
                words = line.strip().split(' ')
                for idy, word in enumerate(words):
                    word = word.lower()
                    clear_word = word
                    for item in no_chars:
                        clear_word = clear_word.replace(item, '')
                    if word[0] == '_' and word[-1] == '_':
                        continue
                    if clear_word:
                        if clear_word not in self.test_dict:
                            self.test_dict[clear_word] = 1
                        else:
                            self.test_dict[clear_word] += 1
        self.sort_test_list = []
        for item in self.test_dict.items():
            self.sort_test_list.append([item[0], item[1]])
        self.sort_test_list = sorted(self.sort_test_list, key = lambda x : x[1], reverse=True)
        self.test_dict = {}
        for idx, item in enumerate(self.sort_test_list):
            self.test_dict[item[0]] = item[1]
        print('test_dict size: %8d' % len(self.sort_test_list))
        
    def BuildDictFromTrain(self, train_path,lb,ub):
        print('build dictionary from '+train_path)
        no_chars = ['(', ')', '[', ']' , ',', '.','!','?','*','"']
        total_train_word = 0
        total_both_word = 0
        self.train_dict = {}
        self.train_sentences = []
        with open(train_path,'r') as fin:
            for idx, line in enumerate(fin):
                if (idx+1) % 200000 == 0: # show progress
                    print('progress at %8d' % (idx+1))
                words = line.strip().split(' ')
                new_sentence = []
                for word in words:
                    word = word.lower()
                    clear_word = word
                    total_train_word += 1
                    # remove useless characters
                    for item in no_chars: 
                        clear_word = clear_word.replace(item, '').lower()
                    # check clear_word in test_dict
                    if clear_word in test_dict: 
                        word_in_test = test_dict[clear_word]
                        total_both_word += 1
                    else:
                        word_in_test = 0
                    # add into self.train_dict
                    if clear_word in self.train_dict:
                        self.train_dict[clear_word][1] += 1
                    else:
                        self.train_dict[clear_word] = [None,1,word_in_test]
                    new_sentence.append(clear_word)
                self.train_sentences.append(new_sentence)
        train_dict.pop('', None)
        # cut by a self-defined threshold, cutbound
        print('keep all words in test_dict, size='+str(len(self.test_dict)))
        print('filter train_dict by count, lower bound='+str(lb)+' and upper bound='+str(ub))
        self.sort_train_list = []
        for key,value in self.train_dict.items():
            if key in self.test_dict:
                self.sort_train_list.append([key,value[1]])
            else:
                if (value[1] >= lb)& (value[1] <= ub):
                    self.sort_train_list.append([key, value[1]])
        self.sort_train_list = sorted(sort_train_list, key = lambda x : x[1], reverse=True)
        self.train_dict = {}
        for idx, item in enumerate(sort_train_list):
            self.train_dict[item[0]] = item[1]
        print('total train dict size     : %8d' % len(sort_train_list))
        print('total train_sentences size: %8d' % len(self.train_sentences))
        print('total_train_word          : %8d' % total_train_word)
        print('total_both_word           : %8d' % total_both_word)
        print('train_dict size           : %8d' % len(self.train_dict))
        
    def BothDict(self):
        self.both_dict = {}
        for key,value in self.train_dict.items():
            train_count = self.train_dict[key]
            if key in test_dict:
                test_count = self.test_dict[key]
            else:
                test_count = 0
            self.both_dict[key] = [None,train_count,test_count]