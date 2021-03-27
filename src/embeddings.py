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
import torch.nn as nn
import torch
from gensim.models import KeyedVectors


class Embedding(object):
    def __init__(self, params, path='wiki-news-300d-1M.vec'):
        self.path = path
        self.embedding_size = params.embedding_size
    
    def load_embeddings(self, t_words):
        print("Loading pre-trained embeddings...")
        if not os.path.exists(self.path):
            print("Embeddings not found")
        self.word2vec = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
        len_words = len(t_words.word_index) + 1
        self.embedding_weights = np.zeros((len_words, self.embedding_size))
        word2id = t_words.word_index
        for word, index in word2id.items():
            try:
                self.embedding_weights[index, :] = self.word2vec[word]
            except KeyError:
                pass
        print("--Done--")
        self.create_emb_layer()
        print("--Done--")
        return self.emb_layer
        
    def create_emb_layer(self, non_trainable=False):
        print("Creating embedding layer...")
        # import pdb pdb.set_trace()
        self.num_embeddings, self.embedding_dim = self.embedding_weights.shape
        self.emb_layer = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.emb_layer.load_state_dict({'weight': torch.Tensor(self.embedding_weights)})
        if non_trainable:
            self.emb_layer.weight.requires_grad = False

        


