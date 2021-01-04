import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Positional encoding to be used in transfromer model class
    """

    def __init__(self, d_model, seq_length, dim, dropout=0.1, max_len=5000):
        """
        Arguments:
        ----------
            d_model : embedding size
            seq_length : length of events
            dim : num_events possible per day
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # initialize positional encoding
        pe = torch.zeros(max_len, d_model)

        position = torch.tensor(
            [float(i) for i in range(dim)] * seq_length
            + [0.0] * (max_len - seq_length * dim),
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # standard positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    
    
class TransformerCNNModel(nn.Module):
    def __init__(
        self,
        ntoken,
        ninp,
        nhead,
        nhid,
        nlayers,
        num_classes,
        device="cpu",
        seq_length=366,
        num_events=10,
        dropout=0.5,
    ):
        """
        Initialize a transformer model for hospital readmissions. The model consists of the following:
        - Transformer encoder layers
        - Single 1D CNN layer
        - Final fully connected layer to determine probability of readmissions
        
        Args:
        -----
            ntoken : number of tokens in embedding layer (vocabulary size)
            ninp : embedding dimension (number of inputs)
            
            nhead : number of heads in transformers
            nhid : number of transformer linear dimensions
            
            nlayers : number of layers in transfromer
            
            num_classes : number of classes to predict (in this case, binary)
            
            seq_length : length of sequence in batched data
            num_events : maximum number of events per day
            
            dropout : strength of regularization
        """
        super(TransformerCNNModel, self).__init__()
        self.model_type = "Transformer"
        self.device_type = device
        self.emsize = ninp
        self.nhead = nhead
        self.nlayers = nlayers
        
        print(
            "parameters: embsize:{}, nhead:{}, nhid:{}, nlayers:{}, dropout:{}".format(
                ninp, nhead, nhid, nlayers, dropout
            )
        )

        # Inputs into transformer: positional encoding and embeddings
        self.pos_encoder = PositionalEncoding(ninp, seq_length, num_events, dropout)
        self.seq_emb = nn.Embedding(ntoken, ninp)

        # Transformer layer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # CNN & fully connected layers
        self.ff = nn.Linear(int(seq_length) * num_events, int(seq_length))
        self.fc = nn.Linear(int(seq_length), num_classes)
        self.softmax = nn.Softmax(-1)
        self.Conv1d = nn.Conv1d(ninp, 1, 1, stride=1)

        # record
        self.ninp = ninp
        self.dropout = dropout
        self.num_events = num_events
        self.num_classes = num_classes

        # initalize weights
        self.init_weights()

    def seq_embedding(self, seq):
        """Convert the sequence of events into embedding vectors, into single row per observation"""
        batch, length_seq, dim = seq.size()
        seq = seq.contiguous().view(batch * length_seq, dim)

        seq = self.seq_emb(seq)

        seq = seq.contiguous().view(batch, -1, self.ninp)

        return seq

    def init_weights(self):
        """Initialize weights in embedding and fully connected layers"""
        initrange = 0.1

        self.seq_emb.weight.data.uniform_(-initrange, initrange)

        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        self.ff.bias.data.zero_()
        self.ff.weight.data.uniform_(-initrange, initrange)

    def forward0(self, src, mask=None):
        """
        Forward propagation steps:
        - convert events into embedding vectors & positional encoding
        - transformer encoder layers
        - CNN layer
        - final 
        """
        #import pdb; pdb.set_trace()
        # create mask to remove padded entries from calculations for interpretability
        if mask is not None:
            mask = mask.view(mask.size()[0], -1)
            src_mask = mask == 0
            src_mask = src_mask.view(src_mask.size()[0], -1)
            out_mask = (
                mask.float()
                .masked_fill(mask == 0.0, float(-1000.0))
                .masked_fill(mask == 2.0, float(-1000.0))
                .masked_fill(mask == 1.0, float(0.0))
                .view(mask.size()[0], -1)
            )

        src = self.seq_embedding(src).transpose(0, 1) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #print('src', src.shape)
        trans_output = (
            self.transformer_encoder(src, src_key_padding_mask=src_mask)
            .transpose(0, 1)
            .transpose(1, 2)
        )
        #print('#out', trans_output.shape)
        final_feature_map = self.Conv1d(trans_output).squeeze()
        #print(final_feature_map.shape)
    
        # if out_mask is not None:
        # extract normalized feature importances per prediction
        importance_out = self.softmax(final_feature_map + out_mask)

        output = self.ff(final_feature_map)
        output = self.fc(output)

        # ensure no accidental additional dimensions
        if len(output.size()) != 2:
            output = output.view(1, 2)

        return output, importance_out
    

    def forward(self, src, mask=None):
        """
        Forward propagation steps:
        - convert events into embedding vectors & positional encoding
        - transformer encoder layers
        - CNN layer
        - final 
        """
        import pdb; pdb.set_trace()
        # create mask to remove padded entries from calculations for interpretability
        if mask is not None:
            mask = mask.view(mask.size()[0], -1)
            src_mask = mask == 0
            src_mask = src_mask.view(src_mask.size()[0], -1)
            out_mask = (
                mask.float()
                .masked_fill(mask == 0.0, float(-1000.0))
                .masked_fill(mask == 2.0, float(-1000.0))
                .masked_fill(mask == 1.0, float(0.0))
                .view(mask.size()[0], -1)
            )

        src = self.seq_embedding(src).transpose(0, 1) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #print('src', src.shape)
        trans_output = (
            self.transformer_encoder(src, src_key_padding_mask=src_mask)
            .transpose(0, 1)
            .transpose(1, 2)
        )
        #print('#out', trans_output.shape)
        final_feature_map = self.Conv1d(trans_output).squeeze()
        #print(final_feature_map.shape)
    
        # if out_mask is not None:
        # extract normalized feature importances per prediction
        importance_out = self.softmax(final_feature_map + out_mask)

        output = self.ff(final_feature_map)
        output = self.fc(output)

        # ensure no accidental additional dimensions
        if len(output.size()) != 2:
            output = output.view(1, 2)

        return output, importance_out
    

    def forward_shap(self, src, mask=None):
        """
        Forward propagation steps:
        - convert events into embedding vectors & positional encoding
        - transformer encoder layers
        - CNN layer
        - final 
        """
        # create mask to remove padded entries from calculations for interpretability
        if mask is not None:
            mask = mask.view(mask.size()[0], -1)
            src_mask = mask == 0
            src_mask = src_mask.view(src_mask.size()[0], -1)
            out_mask = (
                mask.float()
                .masked_fill(mask == 0.0, float(-1000.0))
                .masked_fill(mask == 2.0, float(-1000.0))
                .masked_fill(mask == 1.0, float(0.0))
                .view(mask.size()[0], -1)
            )

        src = self.seq_embedding(src).transpose(0, 1) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #print('src', src.shape)
        trans_output = (
            self.transformer_encoder(src, src_key_padding_mask=src_mask)
            .transpose(0, 1)
            .transpose(1, 2)
        )
        #print('#out', trans_output.shape)
        final_feature_map = self.Conv1d(trans_output).squeeze()
        #print(final_feature_map.shape)
    
        # if out_mask is not None:
        # extract normalized feature importances per prediction
        importance_out = self.softmax(final_feature_map + out_mask)

        output = self.ff(final_feature_map)
        output = self.fc(output)

        # ensure no accidental additional dimensions
        if len(output.size()) != 2:
            output = output.view(1, 2)

        return output, importance_out