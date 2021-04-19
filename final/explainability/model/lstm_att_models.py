"""LSTM-based with Attention Model Architectures."""


import numpy as np
import torch
import torch.nn as nn

from lstm_models import *


class AttNoHtLSTM(SimpleLSTM):
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
        dropout=None,
        init_type="zero",
        linear_bias=True,
    ):
        super(AttNoHtLSTM, self).__init__(
            emb_dim=emb_dim, hidden_dim=hidden_dim, vocab=vocab, device=device
        )

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
        self.linear_bias = linear_bias

        """
        self.attn_layer = (
            nn.Linear(hidden_dim *2, 1, bias=linear_bias) 
            if bidi else nn.Linear(hidden_dim, 1, bias=linear_bias)
        )
        """
        if dropout is None:
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=nlayers,
                bidirectional=bidi,
                batch_first=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=nlayers,
                bidirectional=bidi,
                batch_first=True,
                dropout=dropout,
            )

        self.pred_layer = (
            nn.Linear(hidden_dim * 2, 1, bias=linear_bias)
            if bidi
            else nn.Linear(hidden_dim, 1, bias=linear_bias)
        )

        self.dpt = nn.Dropout(dropout)

        """
        self.context_layer = (
            nn.Linear(hidden_dim * 2, 1, bias=linear_bias) 
            if bidi else nn.Linear(hidden_dim, 1, bias=linear_bias)
        )
        """
        self.init_weights()

    def forward(self, tokens, ret_attn=False):

        if self.dpt is not None:
            embedded = self.dpt(self.emb_layer(tokens))
        else:
            embedded = self.emb_layer(tokens)

        if self.init_type == "learned":
            self.h0.requires_grad = True
            self.c0.requires_grad = True
            hidden = (
                self.h0.repeat(1, tokens.shape[0], 1),
                self.c0.repeat(1, tokens.shape[0], 1),
            )

        else:  # default behavior
            hidden = self.init_hidden(tokens.shape[0])
            hidden = self.repackage_hidden(hidden)

        text_lengths = torch.sum(tokens != self.pad_idx, dim=1).to("cpu")

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, enforce_sorted=False, batch_first=True
        )

        packed_output, (final_hidden, cell) = self.lstm(packed_embedded, hidden)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=tokens.shape[1]
        )

        if self.bidi:
            out = torch.cat(
                [output[:, -1, : self.hidden_dim], output[:, 0, self.hidden_dim :]],
                dim=1,
            )
        else:
            out = output[:, -1, :]

        # Switch to multiplicative attention
        mask_feats = np.array(tokens.cpu().numpy() == 0)
        mask_feats = -1000 * mask_feats.astype(np.int)

        mask_feats = torch.Tensor(mask_feats).to(self.device)

        attn_weights_int = torch.bmm(output, out.unsqueeze(2)).squeeze(2) / (
            (tokens.shape[1]) ** 0.5
        )
        attn_weights = nn.functional.softmax(attn_weights_int + mask_feats, -1)

        context = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(-1)).squeeze(
            -1
        )

        concat_out = context

        if self.dpt is not None:
            pred = self.pred_layer(self.dpt(concat_out))
        else:
            pred = self.pred_layer(concat_out)

        if ret_attn:
            return (
                pred.detach().cpu().numpy(),
                attn_weights.detach().cpu().numpy(),
                context.detach().cpu().numpy(),
                attn_weights_int.detach().cpu().numpy(),
                out.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
            )

        return pred

    def forward_shap(self, token_ids, mask, full_id_matrix=False):
        token_ids = token_ids if token_ids.is_cuda else token_ids.to(self.device)

        if self.init_type == "learned":
            self.h0.requires_grad = False
            self.c0.requires_grad = False

            hidden = (self.h0.repeat(1, 1, 1), self.c0.repeat(1, 1, 1))

        else:  # default behavior
            hidden = self.init_hidden(1)
            hidden = self.repackage_hidden(hidden)

        token_ids[sum(mask) :, :] = 0
        embedded = torch.matmul(token_ids, self.emb_layer.weight).unsqueeze(0)

        embedded = embedded[:, : sum(mask), :]

        output, _ = self.lstm(embedded, hidden)

        # output = output.permute(1, 0, 2)  # [batch, text_length, hidden_dim]
        # print(f'Output dimensions: {output.shape}')
        if self.bidi:
            out = torch.cat(
                [output[:, -1, : self.hidden_dim], output[:, 0, self.hidden_dim :]],
                dim=1,
            )
        else:
            out = output[:, -1, :]
        # import IPython.core.debugger

        # dbg = IPython.core.debugger.Pdb()
        # dbg.set_trace()

        # print(f'Stacked hidden dimensions: {stacked_hidden.shape}')
        # print(f'mask weight dimensions: {mask_feats.shape}')
        # attention = self.context_layer(output).squeeze(-1)
        # att_weights = nn.functional.softmax(attention, dim=-1)
        # context = torch.bmm(att_weights.unsqueeze(1), output).squeeze(1)
        attn_weights = torch.bmm(output, out.unsqueeze(2)).squeeze(2) / (
            sum(mask) ** 0.5
        )

        soft_attn_weights = nn.functional.softmax(attn_weights, 1)

        context = torch.bmm(
            output.transpose(1, 2), soft_attn_weights.unsqueeze(-1)
        ).squeeze(-1)

        # concat_out = torch.cat((context, out), dim=1)
        concat_out = context
        pred = self.pred_layer(concat_out)

        return pred

