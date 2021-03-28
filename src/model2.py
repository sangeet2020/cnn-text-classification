import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import Embedding

class TextClassifier(nn.ModuleList):
    def __init__(self, t_words, params):
        super(TextClassifier, self).__init__()

		# Parameters regarding text preprocessing
        self.t_words = t_words
        self.max_seq_len = params.max_seq_len
        self.num_words = params.num_words
        self.embedding_size = params.embedding_size
        
        
        D = params.embedding_size
        V = len(self.t_words.word_index) + 1
        C = 20
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]

        # self.embed = nn.Embedding(V, D, padding_idx=0)
        self.embed = Embedding(params).load_embeddings(self.t_words)
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (int(K), D)) for K in Ks])
        self.dropout = nn.Dropout(params.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        
        self.embed.weight.requires_grad = False    
    
    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit