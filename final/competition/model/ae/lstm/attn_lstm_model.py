"""
Attention + RNN model definition
- embedding layer
- LSTM
- fully connected layers
- Attention layer

Author: Lin Lee Cheong
Start: 1/29/2020
Last updated: 2/3/2020


Assume single layer LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        bidirectional,
        dropout,
        padding_idx=None,
        device=None,
    ):
        super().__init__()

        self.model_type = "LSTM-GeneralAttention"
        self.vocab_size = vocab_size
        self.embedding_dim = (embedding_dim,)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.dropout_probs = dropout
        self.padding_idx = padding_idx
        self.device = device if device is not None else "cpu"

        # TODO: Added
        self.batch_size = 2048

        # define model layers
        self.embedding = (
            nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            if padding_idx is not None
            else nn.Embedding(vocab_size, embedding_dim)
        )

        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=bidirectional, dropout=dropout
        )

        self.fc = (
            nn.Linear(hidden_dim * 2, output_dim)
            if bidirectional
            else nn.Linear(hidden_dim, output_dim)
        )

        """
        self.fc1 = (nn.Linear(hidden_dim * 2, 4 * output_dim) if bidirectional
                    else nn.Linear(hidden_dim, 4 * output_dim))
        self.fc2 = (nn.Linear( 4 * output_dim, output_dim) if bidirectional
                    else nn.Linear( 4 * output_dim, output_dim))
        self.relu = nn.ReLU() 
        """
        # regularization
        self.dropout = nn.Dropout(dropout)

        # initialize
        self.init_weights()

    def attention_layer(self, output, final_hidden, explain=False, mask=None):
        """Dot Attention"""
        stacked_hidden = (
            torch.cat((final_hidden[0], final_hidden[1]), dim=1)
            if self.bidirectional
            else final_hidden.squeeze(0)
        )

        # print(f'Stacked hidden dimensions: {stacked_hidden.shape}')
        stacked_hidden = stacked_hidden.unsqueeze(2)

        # print(f'Added third dimension: {stacked_hidden.shape}')
        attn_weights = torch.bmm(output, stacked_hidden).squeeze(2)
        # print(f'is cuda:{attn_weights.is_cuda}')

        # print(f'Attention weight dimensions: {attn_weights.shape}')
        # add padding masks similar to BERT implementation

        if mask is not None:
            attn_weights += mask

        # batch_sizexnum_featuresx1
        soft_attn_weights = (F.softmax(attn_weights, 1)).unsqueeze(2)
        # print(f'Soft attention weight dimensions: {soft_attn_weights.shape}')
        # print(f'Output dimensions: {output.shape}')

        # print(f'transpose: {output.transpose(1, 2).shape}')
        new_stacked_hidden = (
            torch.bmm(output.transpose(1, 2), soft_attn_weights)
        ).squeeze(2)
        # print(f'New stacked hidden dimensions: {new_stacked_hidden.shape}')

        if self.bidirectional:
            new_hidden = torch.stack(
                (
                    new_stacked_hidden[:, : self.hidden_dim],
                    new_stacked_hidden[:, self.hidden_dim :],
                )
            )
        else:
            new_hidden = new_stacked_hidden.unsqueeze(0)

        # print(f'New  hidden dimensions: {new_hidden.shape}')

        if explain:
            return new_hidden, soft_attn_weights

        return new_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        directionality = 2 if self.bidirectional else 1
        return (
            weight.new_zeros(directionality, batch_size, self.hidden_dim),
            weight.new_zeros(directionality, batch_size, self.hidden_dim),
        )

    def repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.

        Needed to prevent RNN+Attention backpropagating between batches.
        """

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    # def forward(self, text, text_lengths, hidden, explain=False):
    def forward(self, text, text_lengths=None, batch_size=2048, explain=False):
        # TODO: Added
        h = self.init_hidden(batch_size)
        # hidden = self.repackage_hidden(h)
        if isinstance(h, torch.Tensor):
            hidden = h.detach()
        else:
            hidden = tuple(self.repackage_hidden(v) for v in h)
        text_lengths = torch.tensor(np.ones([batch_size]) * 500)
        # import pdb; pdb.set_trace()
        ##End TODO

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, enforce_sorted=False, batch_first=True
        )
        packed_output, (hidden, cell) = self.rnn(packed_embedded, hidden)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = output.permute(1, 0, 2)  # [batch, text_length, hidden_dim]

        mask_feats = np.array(text.cpu().numpy() == 0)
        mask_feats = -500 * mask_feats.astype(np.int)
        mask_feats = torch.Tensor(mask_feats).to(self.device)

        if explain:
            hidden, soft_attn_weights = self.attention_layer(
                output, hidden, explain=True, mask=mask_feats
            )
        else:
            hidden = self.attention_layer(output, hidden, mask=mask_feats)

        # print(f'Returned  hidden dimensions: {hidden.shape}')
        fhidden = (
            self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            if self.bidirectional
            else self.dropout(hidden[-1, :, :])
        )

        """
        out = self.dropout(self.fc1(fhidden))
        out = self.fc2(self.relu(out))
                    
        if explain:
            return out, fhidden, soft_attn_weights
        return out, fhidden
        """
        if explain:
            return self.fc(fhidden), fhidden, soft_attn_weights

        return self.fc(fhidden)  # , fhidden #TODO: Modified

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

        self.fc.weight.data.uniform_(-initrange * 0.1, initrange * 0.1)
        self.fc.bias.data.zero_()
        """
        self.fc1.weight.data.uniform_(-initrange*.1, initrange*.1)
        self.fc1.bias.data.zero_()   
        self.fc2.weight.data.uniform_(-initrange*.1, initrange*.1)
        self.fc2.bias.data.zero_()   
        """
