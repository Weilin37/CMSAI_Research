"""LSTM-based Model Architectures for LRP."""

import numpy as np
from numpy import newaxis as na
import torch
import torch.nn as nn

from lstm_models import *


class LSTM_LRP_MultiLayer(nn.Module):
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

