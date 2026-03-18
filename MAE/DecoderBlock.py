from torch import nn
from FFN import PositionWiseFFN
from Multi_Head_Attention import MultiHeadAttention
from rresidual_layer_normalization import AddNorm

class MAEDecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, **kwargs):
        super(MAEDecoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, x, state=None):
        x1 = self.attention(x, x, x, valid_lens=None)
        y = self.addnorm1(x, x1)
        z = self.addnorm2(y, self.ffn(y))
        return z, state