#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
<Function of script>
"""

import os
import sys
import argparse
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
    
    def load_data(self):
        self.newsgroups_data = fetch_20newsgroups(subset='all')
        
    def get_targets(self):
        self.targets = self.newsgroups_data.target
    
    def get_sentences(self):
        self.sentences = self.newsgroups_data.data
    
    def get_id2target(self):
        self.id2target = {}
        uniq_targets = list(self.newsgroups_data.target_names)
        for idx, cat in enumerate(uniq_targets):
            self.id2target[idx] = cat
    
    def targets2one_hot(self):
        # self.y = to_categorical(self.targets)
        self.y = self.targets
        
    def tokenize(self):
        self.t_words = Tokenizer()
        self.t_words.fit_on_texts(self.sentences)  
        self.X = self.t_words.texts_to_sequences(self.sentences)
    
    def build_vocabulary(self):
        self.vocabulary = self.t_words.word_index
        
    def padding(self, truncating='post'):
        self.X = pad_sequences(self.X, maxlen=self.max_seq_len, truncating='post')
    
    def split_data(self):
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, shuffle=True, 
                                                                                test_size=0.2, random_state=123) 
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, 
                                                                                            shuffle=True, test_size=0.25, 
                                                                                            random_state=123)
