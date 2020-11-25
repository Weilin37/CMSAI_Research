"""
Basic RNN model definition
- embedding layer
- LSTM
- fully connected layers

Author: Lin Lee Cheong
Start: 1/29/2020
Last updated: 1/29/2020
"""
import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim,
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_probs = dropout
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = (nn.Linear(hidden_dim * 2, output_dim) if bidirectional 
                   else nn.Linear(hidden_dim, output_dim))
        
        # regularization
        self.dropout = nn.Dropout(dropout)
        
        # initialize
        self.init_weights()
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, enforce_sorted=False, batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
       
        hidden = (self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                  if self.bidirectional 
                  else self.dropout(hidden[-1,:,:], dim = 1))
        
        return self.fc(hidden)
    
    def init_weights(self):
        initrange = 0.5
        
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        self.fc.weight.data.uniform_(-initrange*.1, initrange*.1)
        
        self.fc.bias.data.zero_()   