"""LSTM-based Model Architectures."""

import numpy as np
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        vocab,
        device,
        nlayers=1,
        bidi=True,
        use_gpu=True,
        pad_idx=0,
    ):
        super(SimpleLSTM, self).__init__()

        self.device = device
        self.use_gpu = use_gpu

        self.emb_dim = emb_dim
        self.input_dim = len(vocab)
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.emb_layer = nn.Embedding(self.input_dim, emb_dim, padding_idx=pad_idx)

        self.hidden_dim = hidden_dim
        self.bidi = bidi
        self.nlayers = nlayers
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=nlayers,
            bidirectional=bidi,
            batch_first=True,
        )

        self.pred_layer = (
            nn.Linear(hidden_dim * 2, 2) if bidi else nn.Linear(hidden_dim, 2)
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.3
        self.emb_layer.weight.data.uniform_(-initrange, initrange)

        self.pred_layer.weight.data.uniform_(-initrange, initrange)
        self.pred_layer.bias.data.zero_()

    def init_hidden(self, batch_size):
        count = self.nlayers * 2 if self.bidi else self.nlayers

        weight = next(self.parameters())

        directionality = 2 if self.bidi else 1
        weights = (
            weight.new_zeros(count * directionality, batch_size, self.hidden_dim),
            weight.new_zeros(count * directionality, batch_size, self.hidden_dim),
        )

        if self.use_gpu:
            return (weights[0].cuda(), weights[1].cuda())

        return weights

    def repackage_hidden(h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        Needed to prevent RNN+Attention backpropagating between batches.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()

        if isinstance(h, tuple):
            return (v.detach() for v in h)

    def forward(self, tokens):
        embedded = self.emb_layer(tokens)

        hidden = self.init_hidden(tokens.shape[0])
        hidden = self.repackage_hidden()

        text_lengths = torch.sum(tokens != self.pad_idx, dim=1).to("cpu")

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, enforce_sorted=False, batch_first=True
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded, hidden)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = output.permute(1, 0, 2)  # [batch, text_length, hidden_dim]

        if self.bidi:
            out = torch.cat(
                [output[:, -1, : self.hidden_dim], output[:, 0, self.hidden_dim :]],
                dim=1,
            )
        else:
            out = output[:, -1, :]

        pred = self.pred_layer(out)

        return pred

    def save_model(self, filepath):
        if not os.path.isdir(os.path.split(filepath)[0]):
            os.makedirs(os.path.split(filepath)[0])
        torch.save({self.state_dict()}, filepath)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def get_ids_masks(self, token_list, padding_length):
        """Only enabled for one example at a time"""
        ids, masks = np.zeros((padding_length, len(self.vocab))), [0] * padding_length

        for (i, idx_token) in enumerate(token_list):
            if i >= padding_length or idx_token == self.pad_idx:
                break
            ids[i, idx_token] = 1
            masks[i] = 1

        return torch.FloatTensor(ids), masks

    def get_all_ids_masks(self, data, padding_length):
        ids, masks = [], []
        for obs in data:
            id_vals, mask_vals = self.get_ids_masks(obs, padding_length)
            ids.append(id_vals)
            masks.append(mask_vals)

        return torch.stack(ids), masks

    def forward_shap(self, token_ids, mask, full_id_matrix=False):
        """Only enabled for one example at a time"""
        token_ids = token_ids if token_ids.is_cuda else token_ids.cuda()

        hidden = self.init_hidden(1)
        hidden = self.repackage_hidden()

        emb = torch.matmul(token_ids, self.emb_layer.weight).unsqueeze(0)

        out, _ = self.lstm(emb, hidden)

        return self.pred_layer(out).squeeze(0)
