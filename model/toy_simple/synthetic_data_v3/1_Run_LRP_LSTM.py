#!/usr/bin/env python
# coding: utf-8

# ## Run LRP on all test and validation results

# In[3]:


import sys

sys.path.append("../")
sys.path.append("../../")

import os
import json
import time
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from urllib.parse import urlparse
import tarfile
import pickle
import shutil
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import numpy as np
from numpy import newaxis as na

import deep_id_pytorch

from lstm_models import *
from att_lstm_models import *
from lstm_utils import *
from xgboost_utils import *
#from lrp_att_model import *
import shap_jacc_utils as sj_utils

import rbo

MODEL_NAME = "lstm-att-lrp"

NROWS = 1e9

TRAIN_MODEL = True
SEQ_LEN = 30
DATA_TYPE = 'seq_based'

#XGB_BEST_SHAP_PATH = f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/xgb/shap/test_shap_15.pkl"

TRAIN_DATA_PATH = f"../../../data/toy_dataset/data/final_final/{DATA_TYPE}/{SEQ_LEN}/train.csv"
VALID_DATA_PATH = f"../../../data/toy_dataset/data/final_final/{DATA_TYPE}/{SEQ_LEN}/val.csv"
TEST_DATA_PATH = f"../../../data/toy_dataset/data/final_final/{DATA_TYPE}/{SEQ_LEN}/test.csv"
SELECTED_EXAMPLES_PATH = f"../../../data/toy_dataset/data/final_final/{DATA_TYPE}/{SEQ_LEN}/visualized_test_patients.txt"
VOCAB_PATH = f"../../../data/toy_dataset/data/final_final/{DATA_TYPE}/{SEQ_LEN}/vocab.pkl"

MODEL_SAVE_PATH_PATTERN = (
    f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/model_weights/model_{'{}'}.pkl"
)
SHAP_SAVE_DIR_PATTERN = f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/shap/{'{}'}_shap_{'{}'}.pkl"  # SHAP values path for a given dataset split

OUTPUT_RESULTS_PATH = (
    f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/train_results/results.csv"
)
PARAMS_PATH = (
    f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/train_results/model_params.json"
)

PLOT_SAVE_DIR = (
    f"./output/final_final/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/plots/"
)

BEST_EPOCH = 8
TARGET_COLNAME = "label"
UID_COLNAME = "patient_id"
TARGET_VALUE = "1"


# Model Parameters
MODEL_PARAMS = {
    # Dataset/vocab related
    "min_freq": 1,
    "batch_size": 64,
    "num_eval_val": 700,
    "num_eval_test": 7000,
    # Model related parameters
    "embedding_dim": 8,
    "hidden_dim": 16,
    "nlayers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    "linear_bias": False,
    "init_type": "zero",  # zero/learned
    "learning_rate": 0.01,#0.01,
    "scheduler_step": 3,
    "clip": False,
    "rev": False,
    # SHAP-related parameters
    "n_background": 300,  # Number of background examples
    "background_negative_only": True,  # If negative examples are used as background
    "background_positive_only": False,
    "test_positive_only": False,
    "is_test_random": False,
    "n_valid_examples": 64,  # Number of validation examples to be used during shap computation
    "n_test_examples": 64,  # Number of the final test examples to be used in shap computation #TODO
}
def get_wtau(x, y):
    return stats.weightedtau(x, y, rank=None)[0]


def get_rbo(x, y, uid, p=0.7):
    x_idx = np.argsort(x)[::-1]
    y_idx = np.argsort(y)[::-1]

    return rbo.RankingSimilarity(
        [uid[idx] for idx in x_idx], [uid[idx] for idx in y_idx]
    ).rbo(p=p)


# calculate ground truth scores
def is_value(x):
    if "_N" in x:
        return False
    return True


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


class LSTM_LRP_MultiLayer:
    def __init__(self, pymodel):
        super(LSTM_LRP_MultiLayer, self).__init__()

        self.init_model(pymodel)

    def init_model(self, pymodel):

        self.device = pymodel.device
        self.use_gpu = pymodel.use_gpu
        self.bidi = pymodel.bidi

        self.emb_dim = pymodel.emb_dim
        self.vocab = pymodel.vocab
        self.input_dim = len(self.vocab)
        self.pad_idx = pymodel.pad_idx
        self.hidden_dim = pymodel.hidden_dim

        self.emb = pymodel.emb_layer.weight.detach().numpy()

        param_list = list(pymodel.lstm.named_parameters())
        param_dict = {}
        for param_tuple in param_list:
            param_dict[param_tuple[0]] = param_tuple[-1].detach().numpy()

        # rearrange, pytorch uses ifgo format, need to move to icfo/igfo format
        idx_list = (
            list(range(0, self.hidden_dim))
            + list(range(self.hidden_dim * 2, self.hidden_dim * 3))
            + list(range(self.hidden_dim, self.hidden_dim * 2))
            + list(range(self.hidden_dim * 3, self.hidden_dim * 4))
        )
        self.nlayers = pymodel.nlayers

        # i (input), g (candidate), f (forget), o (output) order
        # (4 * hidden_dim, emb_dim)
        self.Wxh_Left = {}
        self.bxh_Left = {}
        self.Whh_Left = {}
        self.bhh_Left = {}

        if self.bidi:
            self.Wxh_Right = {}
            self.bxh_Right = {}
            self.Whh_Right = {}
            self.bhh_Right = {}

        for layer in range(self.nlayers):
            self.Wxh_Left[layer] = param_dict[f"weight_ih_l{layer}"][idx_list]
            self.bxh_Left[layer] = param_dict[f"bias_ih_l{layer}"][idx_list]  # shape 4d
            self.Whh_Left[layer] = param_dict[f"weight_hh_l{layer}"][
                idx_list
            ]  # shape 4d*d
            self.bhh_Left[layer] = param_dict[f"bias_hh_l{layer}"][idx_list]  # shape 4d

            if self.bidi:
                # LSTM right encoder
                self.Wxh_Right[layer] = param_dict[f"weight_ih_l{layer}_reverse"][
                    idx_list
                ]
                self.bxh_Right[layer] = param_dict[f"bias_ih_l{layer}_reverse"][
                    idx_list
                ]
                self.Whh_Right[layer] = param_dict[f"weight_hh_l{layer}_reverse"][
                    idx_list
                ]
                self.bhh_Right[layer] = param_dict[f"bias_hh_l{layer}_reverse"][
                    idx_list
                ]

        # START ADDED: CONTEXT LAYER INIT
        # linear output layer: shape C * 4d
        # 0-d: fwd & context
        # d-2d: rev & context
        # 2d-3d: fwd & final hidden
        # 3d-4d: rev & final hidden
        Why = pymodel.pred_layer.weight.detach().numpy()

        self.Why_Left = Why[:, 2 * self.hidden_dim : 3 * self.hidden_dim]  # shape C*d

        if self.bidi:
            self.Why_Right = Why[:, 3 * self.hidden_dim :]  # shape C*d

        self.Wcy_Left = Why[:, : self.hidden_dim]

        if self.bidi:
            self.Wcy_Right = Why[:, self.hidden_dim : 2 * self.hidden_dim]
        # END ADDED: CONTEXT LAYER INIT

    def set_input(self, tokens):
        T = len(tokens)  # sequence length
        d = int(self.Wxh_Left[0].shape[0] / 4)  # hidden layer dimension
        e = self.emb.shape[1]  # word embedding dimension

        self.w = tokens
        self.x = {}
        self.x_rev = {}
        x = np.zeros((T, e))
        x[:, :] = self.emb[tokens, :]
        self.x[0] = x
        self.x_rev[0] = x[::-1, :].copy()
        self.h_Left = {}
        self.c_Left = {}

        if self.bidi:
            self.h_Right = {}
            self.c_Right = {}

        for layer in range(self.nlayers):
            self.h_Left[layer] = np.zeros((T + 1, d))
            self.c_Left[layer] = np.zeros((T + 1, d))

            if self.bidi:
                self.h_Right[layer] = np.zeros((T + 1, d))
                self.c_Right[layer] = np.zeros((T + 1, d))

        self.att_score = None

    def forward_gate(self, layer, t, idx, idx_i, idx_g, idx_f, idx_o, gate_dir):

        if gate_dir == "left":
            self.gates_xh_Left[layer][t] = np.dot(
                self.Wxh_Left[layer], self.x[layer][t]
            )
            self.gates_hh_Left[layer][t] = np.dot(
                self.Whh_Left[layer], self.h_Left[layer][t - 1]
            )
            self.gates_pre_Left[layer][t] = (
                self.gates_xh_Left[layer][t]
                + self.gates_hh_Left[layer][t]
                + self.bxh_Left[layer]
                + self.bhh_Left[layer]
            )
            self.gates_Left[layer][t, idx] = 1.0 / (
                1.0 + np.exp(-self.gates_pre_Left[layer][t, idx])
            )
            self.gates_Left[layer][t, idx_g] = np.tanh(
                self.gates_pre_Left[layer][t, idx_g]
            )
            self.c_Left[layer][t] = (
                self.gates_Left[layer][t, idx_f] * self.c_Left[layer][t - 1]
                + self.gates_Left[layer][t, idx_i] * self.gates_Left[layer][t, idx_g]
            )
            self.h_Left[layer][t] = self.gates_Left[layer][t, idx_o] * np.tanh(
                self.c_Left[layer][t]
            )

        if gate_dir == "right":
            self.gates_xh_Right[layer][t] = np.dot(
                self.Wxh_Right[layer], self.x_rev[layer][t]
            )
            self.gates_hh_Right[layer][t] = np.dot(
                self.Whh_Right[layer], self.h_Right[layer][t - 1]
            )
            self.gates_pre_Right[layer][t] = (
                self.gates_xh_Right[layer][t]
                + self.gates_hh_Right[layer][t]
                + self.bxh_Right[layer]
                + self.bhh_Right[layer]
            )
            self.gates_Right[layer][t, idx] = 1.0 / (
                1.0 + np.exp(-self.gates_pre_Right[layer][t, idx])
            )
            self.gates_Right[layer][t, idx_g] = np.tanh(
                self.gates_pre_Right[layer][t, idx_g]
            )
            self.c_Right[layer][t] = (
                self.gates_Right[layer][t, idx_f] * self.c_Right[layer][t - 1]
                + self.gates_Right[layer][t, idx_i] * self.gates_Right[layer][t, idx_g]
            )
            self.h_Right[layer][t] = self.gates_Right[layer][t, idx_o] * np.tanh(
                self.c_Right[layer][t]
            )

    def forward_lrp(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T = len(self.w)
        d = int(self.Wxh_Left[0].shape[0] / 4)

        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(
            int
        )  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = (
            np.arange(0, d),
            np.arange(d, 2 * d),
            np.arange(2 * d, 3 * d),
            np.arange(3 * d, 4 * d),
        )  # indices of gates i,g,f,o separately

        # initialize
        self.gates_xh_Left = {}
        self.gates_hh_Left = {}
        self.gates_pre_Left = {}
        self.gates_Left = {}

        if self.bidi:
            self.gates_xh_Right = {}
            self.gates_hh_Right = {}
            self.gates_pre_Right = {}
            self.gates_Right = {}

        for layer in range(self.nlayers):
            self.gates_xh_Left[layer] = np.zeros((T, 4 * d))
            self.gates_hh_Left[layer] = np.zeros((T, 4 * d))
            self.gates_pre_Left[layer] = np.zeros((T, 4 * d))  # gates pre-activation
            self.gates_Left[layer] = np.zeros((T, 4 * d))  # gates activation

            if self.bidi:
                self.gates_xh_Right[layer] = np.zeros((T, 4 * d))
                self.gates_hh_Right[layer] = np.zeros((T, 4 * d))
                self.gates_pre_Right[layer] = np.zeros((T, 4 * d))
                self.gates_Right[layer] = np.zeros((T, 4 * d))

        # START ADDED: INITIALIZE CONTEXT LAYERS
        self.ctxt_Left = np.zeros((1, d))
        self.ctxt_Right = np.zeros((1, d))
        self.att_wgt_Left = np.zeros((T, 1))
        self.att_wgt_Right = np.zeros((T, 1))
        self.att_score = np.zeros((T, 1))

        # END ADDED: INITIALIZE CONTEXT LAYERS

        # START EDIT: cycle through first layer first
        layer = 0
        for t in range(T):
            self.forward_gate(
                layer, t, idx, idx_i, idx_g, idx_f, idx_o, gate_dir="left"
            )
            if self.bidi:
                self.forward_gate(
                    layer, t, idx, idx_i, idx_g, idx_f, idx_o, gate_dir="right"
                )

        # go through all the rest of the layers
        if self.nlayers > 1:
            ## TODO: fix init t-1 (zero time step) Zeroes!!
            self.x[layer + 1] = (
                np.concatenate(
                    (self.h_Left[layer][:T], self.h_Right[layer][:T][::-1]), axis=1
                )
                if self.bidi
                else self.h_Left[layer][:T]
            )

            self.x_rev[layer + 1] = self.x[layer + 1][::-1].copy()

            for layer in range(1, self.nlayers):
                for t in range(T):
                    self.forward_gate(
                        layer, t, idx, idx_i, idx_g, idx_f, idx_o, gate_dir="left"
                    )
                    if self.bidi:
                        self.forward_gate(
                            layer, t, idx, idx_i, idx_g, idx_f, idx_o, gate_dir="right"
                        )

                    self.x[layer + 1] = np.concatenate(
                        (self.h_Left[layer][:T], self.h_Right[layer][:T][::-1]), axis=1
                    )
                    self.x_rev[layer + 1] = self.x[layer + 1][::-1].copy()

        # calculate attention layer & context layer
        top_layer = self.nlayers - 1
        self.att_wgt_Left = np.dot(
            self.h_Left[top_layer][:T, :], self.h_Left[top_layer][T - 1]
        )
        self.att_wgt_Right = np.dot(
            self.h_Right[top_layer][:T, :], self.h_Right[top_layer][T - 1]
        )
        self.att_score = self.stable_softmax(
            (self.att_wgt_Left + self.att_wgt_Right) / (T ** 0.5)
        )

        self.ctxt_Left = (self.att_score[:, na] * self.h_Left[top_layer][:T]).sum(
            axis=0
        )
        self.ctxt_Right = (self.att_score[:, na] * self.h_Right[top_layer][:T]).sum(
            axis=0
        )

        # CALCULATE WITH CONTEXT & OUT, NOT JUST HIDDEN
        # self.y_Left = np.dot(self.Why_Left, self.h_Left[top_layer][T - 1])
        self.y_Left = np.dot(self.Wcy_Left, self.ctxt_Left)

        # self.y_Right = np.dot(self.Why_Right, self.h_Right[top_layer][T - 1])
        self.y_Right = np.dot(self.Wcy_Right, self.ctxt_Right)

        self.s = self.y_Left + self.y_Right

        return self.s.copy()  # prediction scores

    def stable_softmax(self, x):
        z = x - np.max(x)
        num = np.exp(z)
        denom = np.sum(num)
        softmax_vals = num / denom

        return softmax_vals

    def lrp_left_gate(
        self,
        Rc_Left,
        Rh_Left,
        Rg_Left,
        Rx,
        layer,
        t,
        d,
        ee,
        idx,
        idx_f,
        idx_i,
        idx_g,
        idx_o,
        eps,
        bias_factor,
    ):

        # import IPython
        # dbg = IPython.core.debugger.Pdb()
        # dbg.set_trace()

        Rc_Left[layer][t] += Rh_Left[layer][t]
        Rc_Left[layer][t - 1] += lrp_linear(
            self.gates_Left[layer][t, idx_f] * self.c_Left[layer][t - 1],
            np.identity(d),
            np.zeros((d)),
            self.c_Left[layer][t],
            Rc_Left[layer][t],
            2 * d,
            eps,
            bias_factor,
            debug=False,
        )

        Rg_Left[layer][t] += lrp_linear(
            self.gates_Left[layer][t, idx_i] * self.gates_Left[layer][t, idx_g],
            np.identity(d),
            np.zeros((d)),
            self.c_Left[layer][t],
            Rc_Left[layer][t],
            2 * d,
            eps,
            bias_factor,
            debug=False,
        )

        Rx[layer][t] += lrp_linear(
            self.x[layer][t],
            self.Wxh_Left[layer][idx_g].T,
            self.bxh_Left[layer][idx_g] + self.bhh_Left[layer][idx_g],
            self.gates_pre_Left[layer][t, idx_g],
            Rg_Left[layer][t],
            d + ee,
            eps,
            bias_factor,
            debug=False,
        )

        Rh_Left[layer][t - 1] += lrp_linear(
            self.h_Left[layer][t - 1],
            self.Whh_Left[layer][idx_g].T,
            self.bxh_Left[layer][idx_g] + self.bhh_Left[layer][idx_g],
            self.gates_pre_Left[layer][t, idx_g],
            Rg_Left[layer][t],
            d + ee,
            eps,
            bias_factor,
            debug=False,
        )
        return Rc_Left, Rh_Left, Rg_Left, Rx

    def lrp_right_gate(
        self,
        Rc_Right,
        Rh_Right,
        Rg_Right,
        Rx_rev,
        layer,
        t,
        d,
        ee,
        idx,
        idx_f,
        idx_i,
        idx_g,
        idx_o,
        eps,
        bias_factor,
    ):
        Rc_Right[layer][t] += Rh_Right[layer][t]
        Rc_Right[layer][t - 1] += lrp_linear(
            self.gates_Right[layer][t, idx_f] * self.c_Right[layer][t - 1],
            np.identity(d),
            np.zeros((d)),
            self.c_Right[layer][t],
            Rc_Right[layer][t],
            2 * d,
            eps,
            bias_factor,
            debug=False,
        )
        Rg_Right[layer][t] += lrp_linear(
            self.gates_Right[layer][t, idx_i] * self.gates_Right[layer][t, idx_g],
            np.identity(d),
            np.zeros((d)),
            self.c_Right[layer][t],
            Rc_Right[layer][t],
            2 * d,
            eps,
            bias_factor,
            debug=False,
        )

        Rx_rev[layer][t] += lrp_linear(
            self.x_rev[layer][t],
            self.Wxh_Right[layer][idx_g].T,
            self.bxh_Right[layer][idx_g] + self.bhh_Right[layer][idx_g],
            self.gates_pre_Right[layer][t, idx_g],
            Rg_Right[layer][t],
            d + ee,
            eps,
            bias_factor,
            debug=False,
        )

        Rh_Right[layer][t - 1] += lrp_linear(
            self.h_Right[layer][t - 1],
            self.Whh_Right[layer][idx_g].T,
            self.bxh_Right[layer][idx_g] + self.bhh_Right[layer][idx_g],
            self.gates_pre_Right[layer][t, idx_g],
            Rg_Right[layer][t],
            d + ee,
            eps,
            bias_factor,
            debug=False,
        )
        return Rc_Right, Rh_Right, Rg_Right, Rx_rev

    def lrp(self, w, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(w)
        self.forward_lrp()

        T = len(self.w)
        d = int(self.Wxh_Left[0].shape[0] / 4)
        e = self.emb.shape[1]
        C = self.Why_Left.shape[0]  # number of classes
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(
            int
        )  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = (
            np.arange(0, d),
            np.arange(d, 2 * d),
            np.arange(2 * d, 3 * d),
            np.arange(3 * d, 4 * d),
        )  # indices of gates i,g,f,o separately

        # initialize
        Rx = {}
        Rx_rev = {}
        Rx_all = {}

        Rh_Left = {}
        Rc_Left = {}
        Rg_Left = {}  # gate g only

        if self.bidi:
            Rh_Right = {}
            Rc_Right = {}
            Rg_Right = {}  # gate g only

        for layer in range(self.nlayers):
            Rx[layer] = np.zeros(self.x[layer].shape)
            Rx_rev[layer] = np.zeros(self.x[layer].shape)
            Rx_all[layer] = np.zeros(self.x[layer].shape)

            Rh_Left[layer] = np.zeros((T + 1, d))
            Rc_Left[layer] = np.zeros((T + 1, d))
            Rg_Left[layer] = np.zeros((T, d))  # gate g only

            if self.bidi:
                Rh_Right[layer] = np.zeros((T + 1, d))
                Rc_Right[layer] = np.zeros((T + 1, d))
                Rg_Right[layer] = np.zeros((T, d))  # gate g only

        Rctxt_Left = np.zeros((1, d))
        Rctxt_Right = np.zeros((1, d))

        Rout_mask = np.zeros((C))
        Rout_mask[LRP_class] = 1.0

        # process top most layer first
        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        layer = self.nlayers - 1
        """
        Rh_Left[layer][T - 1] = lrp_linear(
            self.h_Left[layer][T - 1],
            self.Why_Left.T,  # 8d
            np.zeros((C)),
            self.s,
            self.s * Rout_mask,
            4 * d,
            eps,
            bias_factor,
            debug=False,
        )
        if self.bidi:
            Rh_Right[layer][T - 1] = lrp_linear(
                self.h_Right[layer][T - 1],
                self.Why_Right.T,  # 8d
                np.zeros((C)),
                self.s,
                self.s * Rout_mask,
                4 * d,
                eps,
                bias_factor,
                debug=False,
            )
        """
        # ADD CONTEXT CALCULATIONS TO CONTEXT LAYER
        Rctxt_Left = lrp_linear(
            self.ctxt_Left,
            self.Wcy_Left.T,  # 8d
            np.zeros((C)),
            self.s,
            self.s * Rout_mask,
            4 * d,
            eps,
            bias_factor,
            debug=False,
        )
        if self.bidi:
            Rctxt_Right = lrp_linear(
                self.ctxt_Right,
                self.Wcy_Right.T,  # 8d
                np.zeros((C)),
                self.s,
                self.s * Rout_mask,
                4 * d,
                eps,
                bias_factor,
                debug=False,
            )

        # CONTRIBUTION FROM ATTN LAYER
        Rh_Left[layer][T - 1] += lrp_linear(
            self.h_Left[layer][T - 1],
            np.identity((d)),
            np.zeros((d)),
            self.ctxt_Left,
            self.att_score[T - 1] * Rctxt_Left,
            4 * d,
            eps,
            bias_factor,
            debug=False,
        )
        if self.bidi:
            Rh_Right[layer][T - 1] += lrp_linear(
                self.h_Right[layer][T - 1],
                np.identity((d)),
                np.zeros((d)),
                self.ctxt_Right,
                self.att_score[T - 1] * Rctxt_Right,
                4 * d,
                eps,
                bias_factor,
                debug=False,
            )

        ee = e if self.nlayers == 1 else 2 * d
        for t in reversed(range(T)):

            Rc_Left, Rh_Left, Rg_Left, Rx = self.lrp_left_gate(
                Rc_Left,
                Rh_Left,
                Rg_Left,
                Rx,
                layer,
                t,
                d,
                ee,
                idx,
                idx_f,
                idx_i,
                idx_g,
                idx_o,
                eps,
                bias_factor,
            )

            # ATTN Relevance scores
            Rh_Left[layer][t - 1] += lrp_linear(
                self.h_Left[layer][t - 1],
                np.identity((d)),
                np.zeros((d)),
                self.ctxt_Left,
                self.att_score[t - 1] * Rctxt_Left,
                4 * d,
                eps,
                bias_factor,
                debug=False,
            )

            if self.bidi:
                Rc_Right, Rh_Right, Rg_Right, Rx_rev = self.lrp_right_gate(
                    Rc_Right,
                    Rh_Right,
                    Rg_Right,
                    Rx_rev,
                    layer,
                    t,
                    d,
                    ee,
                    idx,
                    idx_f,
                    idx_i,
                    idx_g,
                    idx_o,
                    eps,
                    bias_factor,
                )
                # ATTN Relevance scores for top-most layer
                Rh_Right[layer][t - 1] += lrp_linear(
                    self.h_Right[layer][t - 1],
                    np.identity((d)),
                    np.zeros((d)),
                    self.ctxt_Right,
                    self.att_score[t - 1] * Rctxt_Right,
                    4 * d,
                    eps,
                    bias_factor,
                    debug=False,
                )

        # propagate through remaining layers
        if self.nlayers > 1:
            remaining_layers = list(range(0, self.nlayers - 1))[::-1]
            # print(f"remaining layers: {remaining_layers}")

            # no more attn layer flow back
            for layer in remaining_layers:

                # Sum up all the relevances for each of the inputs in sequence
                Rx_all[layer + 1] = Rx[layer + 1] + Rx_rev[layer + 1][::-1, :]

                ee = e if layer == 0 else 2 * d
                for t in reversed(range(T)):
                    # Rh_Left[layer][t]   += lrp_linear(
                    #    self.h_Left[layer][t], np.identity((d)) ,
                    #    np.zeros((d)), self.h_Left[layer][t], #self.x[layer+1][t, :d],
                    #    Rx_all[layer+1][t, :d],
                    #    d, eps, bias_factor, debug=False)
                    # @@@@@@@@@@@@@@@@@@@@@@@@
                    Rh_Left[layer][t] += Rx_all[layer + 1][t, :d]
                    # @@@@@@@@@@@@@@@@@@@@@@@@
                    Rc_Left, Rh_Left, Rg_Left, Rx = self.lrp_left_gate(
                        Rc_Left,
                        Rh_Left,
                        Rg_Left,
                        Rx,
                        layer,
                        t,
                        d,
                        ee,
                        idx,
                        idx_f,
                        idx_i,
                        idx_g,
                        idx_o,
                        eps,
                        bias_factor,
                    )

                    ### RIGHT +++++++++
                    # Rh_Right[layer][t]   += lrp_linear(
                    #    self.h_Right[layer][t], np.identity((d)) ,
                    #    np.zeros((d)), self.h_Right[layer][t], #self.x_rev[layer+1][::-1, :][t, d:],
                    #    Rx_all[layer+1][t, d:],
                    #    d, eps, bias_factor, debug=False)
                    # @@@@@@@@@@@@@@@@@@@@@@@@
                    Rh_Right[layer][t] += Rx_all[layer + 1][::-1, :][t, d:]
                    if self.bidi:
                        Rc_Right, Rh_Right, Rg_Right, Rx_rev = self.lrp_right_gate(
                            Rc_Right,
                            Rh_Right,
                            Rg_Right,
                            Rx_rev,
                            layer,
                            t,
                            d,
                            ee,
                            idx,
                            idx_f,
                            idx_i,
                            idx_g,
                            idx_o,
                            eps,
                            bias_factor,
                        )

        # record
        self.Rx_all = Rx_all
        self.Rx = Rx
        self.Rx_rev = Rx_rev
        self.Rh_Left = Rh_Left
        self.Rh_Right = Rh_Right
        self.Rc_Left = Rc_Left
        self.Rc_Right = Rc_Right
        self.Rg_Right = Rg_Right
        self.d = d
        self.ee = ee
        self.Rctxt_Left = Rctxt_Left
        self.Rctxt_Right = Rctxt_Right

        return (
            Rx[0],
            Rx_rev[0][::-1, :],
            Rh_Left[0][-1].sum()
            + Rc_Left[0][-1].sum()
            + Rh_Right[0][-1].sum()
            + Rc_Right[0][-1].sum(),
        )

    def get_attn_values(self):
        return self.att_score


def get_sim(idx_model, idx_gt):
    return len(set(idx_model).intersection(set(idx_gt))) / len(idx_gt)


def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = np.where(hout[na, :] >= 0, 1.0, -1.0)  # shape (1, M)

    # numer    = (w * hin[:,na]) + ( (bias_factor*b[na,:]*1.) * 1./bias_nb_units )
    numer = (w * hin[:, na]) + (
        bias_factor * (b[na, :] * 1.0 + eps * sign_out * 1.0) / bias_nb_units
    )  # shape (D, M)

    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

    denom = hout[na, :] + (eps * sign_out * 1.0)  # shape (1, M)

    message = (numer / denom) * Rout[na, :]  # shape (D, M)

    Rin = message.sum(axis=1)  # shape (D,)

    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note:
    # - local  layer   relevance conservation
    #   if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation
    #   if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections
    # -> can be used for sanity check

    return Rin


def get_sub_valid_data(n_val_eval, batch_size, valid_dataloader):
    """Get subset of validation dataset to run SHAP/LRP on"""

    n_loads = int(np.ceil(n_val_eval / batch_size))
    counter = 0

    for ids, labels, idxed_text in valid_dataloader:
        counter += 1

        if counter == 1:
            sub_val_ids, sub_val_labels, sub_val_idxed_text = ids, labels, idxed_text
        else:
            sub_val_ids = sub_val_ids + ids
            sub_val_labels = torch.cat([sub_val_labels, labels])
            sub_val_idxed_text = torch.cat([sub_val_idxed_text, idxed_text])

        if counter == n_loads:
            break

    sub_val_ids = sub_val_ids[:n_val_eval]
    sub_val_labels = sub_val_labels[:n_val_eval]
    sub_val_idxed_text = sub_val_idxed_text[:n_val_eval]

    return (sub_val_ids, sub_val_labels, sub_val_idxed_text)


def glfass_single(cpu_model, background, test, seq_len, device):
    """
    Single-thread function for Get Lstm Features And Shap Scores
    Called by get_lstm_features_and_shap_scores_mp
    """
    start_time = time.time()

    model = cpu_model.to(device)

    try:

        background_ids, background_labels, background_idxes = background
        bg_data, bg_masks = model.get_all_ids_masks(background_idxes, seq_len)

        explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(
            model, bg_data, bg_masks, gpu_memory_efficient=True
        )

        model.train()
        test_ids, test_labels, test_idxes = test
        test_data, test_masks = model.get_all_ids_masks(test_idxes, seq_len)

        #         import pdb

        #         pdb.set_trace()

        lstm_shap_values = explainer.shap_values(
            test_data, test_masks, model_device=device
        )

    except Exception as excpt:
        print(excpt)
        raise Exception
        # import IPython.core.debugger

        # dbg = IPython.core.debugger.Pdb()
        # dbg.set_trace()
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    # print(f"{device}: test_ids={len(test_ids)}, test_labels={len(test_labels)}, test_idxes={len(test_idxes)}")
    # print(f"Completed on {device} taking {mins}:{secs}")
    return (test_ids, test_labels, test_idxes, lstm_shap_values)


def mycallback(x):
    return x


def myerrorcallback(exception):
    print(exception)
    return exception


def get_lstm_features_and_shap_scores_mp(
    model,
    tr_dataloader,
    test,  # don't use dataloader to fix dataset (test_ids, test_labels, test_idxes)
    seq_len,
    shap_path,
    save_output=True,
    n_background=None,
    background_negative_only=False,
    test_positive_only=False,
    is_test_random=False,
    output_explainer=False,
    multigpu_lst=None,  # cuda:1, cuda:2 ...
):
    """Get all features and shape importance scores for each example in te_dataloader."""

    # Get background dataset
    background = sj_utils.get_lstm_background(
        tr_dataloader, n_background=n_background, negative_only=background_negative_only
    )

    # split up test datasets

    n_gpu = len(multigpu_lst)
    gpu_model_tuple = []
    for gpu in multigpu_lst:
        model = copy.deepcopy(model)
        model.device = gpu
        model = model.to(gpu)
        gpu_model_tuple.append((gpu, model))

    # test = sj_utils.get_lstm_data(
    #    te_dataloader,
    #    n_test,
    #    positive_only=test_positive_only,
    #    is_random=is_test_random,
    # )
    test_ids, test_labels, test_idxes = test

    test_labels_lst, test_idxes_lst, test_ids_lst = [], [], []
    n_per_gpu = int(np.ceil(len(test_ids) / n_gpu))
    for idx in range(n_gpu):
        if idx == (n_gpu - 1):
            test_ids_lst.append(test_ids[idx * n_per_gpu :])
            test_labels_lst.append(test_labels[idx * n_per_gpu :])
            test_idxes_lst.append(test_idxes[idx * n_per_gpu :])
        else:
            test_ids_lst.append(test_ids[idx * n_per_gpu : (idx + 1) * n_per_gpu])
            test_labels_lst.append(test_labels[idx * n_per_gpu : (idx + 1) * n_per_gpu])
            test_idxes_lst.append(test_idxes[idx * n_per_gpu : (idx + 1) * n_per_gpu])

    # multiprocess one core one gpu
    # print(f'Starting multiprocess for {n_gpu} cores')
    try:
        from multiprocessing.dummy import Pool as dThreadPool

        pool = dThreadPool(n_gpu)
        # pool = torch.multiprocessing.Pool(n_gpu)  # one feeding each gpu
        func_call_lst = []
        for cur_test_id, cur_test_label, cur_test_idxes, (gpu, model) in zip(
            test_ids_lst, test_labels_lst, test_idxes_lst, gpu_model_tuple
        ):
            # print(f"\nlength of tests={len(cur_test_id)}")
            # print(f"gpu: {n_gpu}")
            # print(f"model: {model.device}")

            func_call = pool.apply_async(
                glfass_single,
                (
                    model.cpu(),
                    background,
                    (cur_test_id, cur_test_label, cur_test_idxes),
                    seq_len,
                    gpu,
                ),
                callback=mycallback,
                error_callback=myerrorcallback,
            )
            func_call_lst.append(func_call)

        # print('Starting to wait')
        for func_call in func_call_lst:
            func_call.wait()

        # print('Collecting results')
        # import IPython.core.debugger
        # dbg = IPython.core.debugger.Pdb()
        # dbg.set_trace()

        test_ids, test_labels, test_idxes, lstm_shap_values = None, None, None, None
        for func_call in func_call_lst:
            init_results = func_call.get()

            # first one
            if test_ids is None:
                test_ids, test_labels, test_idxes, lstm_shap_values = init_results
                test_ids = list(test_ids)
            else:
                test_ids = test_ids + list(init_results[0])
                test_labels = torch.cat([test_labels, init_results[1]], dim=0)
                test_idxes = torch.cat([test_idxes, init_results[2]], dim=0)
                lstm_shap_values = np.concatenate(
                    [lstm_shap_values, init_results[3]], axis=0
                )

    except Exception as excpt:
        print(excpt)
        # raise Exception
    #         import IPython.core.debugger

    #         dbg = IPython.core.debugger.Pdb()
    #         dbg.set_trace()

    finally:
        pool.close()
        pool.join()
        pool.terminate()
        # print('Multiprocessing pool closed')

    # print('collating per patient results')
    try:
        # import IPython.core.debugger
        # dbg = IPython.core.debugger.Pdb()
        # dbg.set_trace()
        test = (test_ids, test_labels, test_idxes)
        features = []
        scores = []
        patients = []
        total = len(test[0])
        import pdb

        # pdb.set_trace()
        for idx in range(total):
            df_shap, patient_id = sj_utils.get_per_patient_shap(
                lstm_shap_values, test, model.vocab, idx
            )
            events = df_shap["events"].values.tolist()
            vals = df_shap["shap_vals"].values.tolist()

            pad = "<pad>"
            if pad in events:
                pad_indx = events.index(pad)
                events = events[:pad_indx]
                vals = vals[:pad_indx]

            features.append(events)
            scores.append(vals[:])
            patients.append(patient_id)

        shap_values = (features, scores, patients)
    except Exception as excpt:
        print(excpt)
        # import pdb

        # pdb.set_trace()
        raise Exception

    if save_output:
        if not os.path.isdir(os.path.split(shap_path)[0]):
            os.makedirs(os.path.split(shap_path)[0])
        save_pickle(shap_values, shap_path)

    if output_explainer:
        return shap_values, explainer.expected_value

    return shap_values


#Load Model Params
with open(PARAMS_PATH, "r") as fp:
    MODEL_PARAMS = json.load(fp)


with open(VOCAB_PATH, "rb") as fp:
    vocab = pickle.load(fp)

valid_dataset, vocab = build_lstm_dataset(
    VALID_DATA_PATH,
    min_freq=MODEL_PARAMS["min_freq"],
    uid_colname=UID_COLNAME,
    target_colname=TARGET_COLNAME,
    max_len=SEQ_LEN,
    target_value=TARGET_VALUE,
    vocab=vocab,
    nrows=NROWS,
    rev=MODEL_PARAMS["rev"],
)
print(f"vocab len: {len(vocab)}")  # vocab + padding + unknown

test_dataset, _ = build_lstm_dataset(
    TEST_DATA_PATH,
    min_freq=MODEL_PARAMS["min_freq"],
    uid_colname=UID_COLNAME,
    target_colname=TARGET_COLNAME,
    max_len=SEQ_LEN,
    target_value=TARGET_VALUE,
    vocab=vocab,
    nrows=NROWS,
    rev=MODEL_PARAMS["rev"],
)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
)

test_dataloader = DataLoader(
    test_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
)


# Best model path

# In[7]:

# Create model

# In[8]:


# Check if cuda is available
print(f"Cuda available: {torch.cuda.is_available()}")
model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[9]:


lstm_model_best = AttNoHtLSTM(
    MODEL_PARAMS["embedding_dim"],
    MODEL_PARAMS["hidden_dim"],
    vocab,
    model_device,
    bidi=MODEL_PARAMS["bidirectional"],
    nlayers=MODEL_PARAMS["nlayers"],
    dropout=MODEL_PARAMS["dropout"],
    init_type=MODEL_PARAMS["init_type"],
    linear_bias=MODEL_PARAMS["linear_bias"],
)

model_path = MODEL_SAVE_PATH_PATTERN.format(str(BEST_EPOCH).zfill(2))
lstm_model_best.load_state_dict(torch.load(model_path))

# In[7]:

def get_eval_data(dataloader, num):
    """Get more than one iteration of data"""
    col_pid, col_lab, col_txt = None, None, None
    col_num = 0
    for pid, lab, txt in dataloader:
        if col_pid is None:
            col_pid, col_lab, col_txt = pid, lab, txt
        else:
            col_pid = tuple(list(col_pid) + list(pid))
            col_lab = torch.cat((col_lab, lab), dim=0)
            col_txt = torch.cat((col_txt, txt), dim=0)
        col_num = len(col_pid)
        if col_num > num:
            break
            
    return col_pid[:num], col_lab[:num], col_txt[:num]    


valid_results_best = {}
valid_results_best[BEST_EPOCH] = {}

test_results_best = {}
test_results_best[BEST_EPOCH] = {}
# calculate relevancy and SHAP
lstm_model_best.eval()
lrp_model = LSTM_LRP_MultiLayer(lstm_model_best.cpu())


val_patient_ids, val_labels, val_idxed_text = get_eval_data(
    valid_dataloader, 7000)

test_patient_ids, test_labels, test_idxed_text = get_eval_data(
    test_dataloader, 7000)

start = time.time()

for sel_idx in range(len(val_labels)):
    one_text = [
        int(token.numpy())
        for token in val_idxed_text[sel_idx]
        if int(token.numpy()) != 0
    ]
    lrp_model.set_input(one_text)
    lrp_model.forward_lrp()

    Rx, Rx_rev, _ = lrp_model.lrp(one_text, 0, eps=1e-6, bias_factor=0)
    R_words = np.sum(Rx + Rx_rev, axis=1)

    df = pd.DataFrame()
    df["lrp_scores"] = R_words
    df["idx"] = one_text
    df["seq_idx"] = [x for x in range(len(one_text))]
    df["token"] = [lstm_model_best.vocab.itos(x) for x in one_text]
    df["att_weights"] = lrp_model.get_attn_values()

    if val_patient_ids[sel_idx] not in valid_results_best[BEST_EPOCH]:
        valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]] = {}
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]] = {}
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["label"] = val_labels[
        sel_idx
    ]
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["imp"] = df.copy()
    
    if sel_idx % 500 == 0:
        print(sel_idx)
        
end = time.time()
mins, secs = epoch_time(start, end)
print(f"{mins}min: {secs}sec")


valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]

save_dir = os.path.dirname(SHAP_SAVE_DIR_PATTERN)
os.makedirs(save_dir, exist_ok=True)

val_results_path = os.path.join(save_dir, f'val_results_lrp_{BEST_EPOCH}.pkl')
with open(val_results_path, 'wb') as fp:
    pickle.dump(valid_results_best, fp)


start = time.time()
for sel_idx in range(len(test_labels)):
    one_text = [
        int(token.numpy())
        for token in test_idxed_text[sel_idx]
        if int(token.numpy()) != 0
    ]
    lrp_model.set_input(one_text)
    lrp_model.forward_lrp()

    Rx, Rx_rev, _ = lrp_model.lrp(one_text, 0, eps=1e-6, bias_factor=0)
    R_words = np.sum(Rx + Rx_rev, axis=1)

    df = pd.DataFrame()
    df["lrp_scores"] = R_words
    df["idx"] = one_text
    df["seq_idx"] = [x for x in range(len(one_text))]
    df["token"] = [lstm_model_best.vocab.itos(x) for x in one_text]
    df["att_weights"] = lrp_model.get_attn_values()

    if test_patient_ids[sel_idx] not in test_results_best[BEST_EPOCH]:
        test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]] = {}
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]] = {}
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["label"] = test_labels[
        sel_idx
    ]
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["imp"] = df.copy()
    
    if sel_idx / 50 == 0:
        print(sel_idx)    
        end = time.time()
        mins, secs = epoch_time(start, end)
        print(f"{mins}min: {secs}sec")
        
end = time.time()
mins, secs = epoch_time(start, end)
print(f"Total {mins}min: {secs}sec")

test_results_path = os.path.join(save_dir, f'test_results_lrp_{BEST_EPOCH}.pkl')
with open(test_results_path, 'wb') as fp:
    pickle.dump(test_results_best, fp)