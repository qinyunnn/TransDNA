import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import einops
import math






class GLU(nn.Module):
    def __init__(self, dim) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Swish(nn.Module):
    def __init__(self, inplace=True) -> None:
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, inputs: Tensor) -> Tensor:
        if self.inplace:
            return inputs.mul_(inputs.sigmoid())
        else:
            return inputs * inputs.sigmoid()


class Relu(nn.Module):
    def __init__(self, inplace=True) -> None:
        super(Relu, self).__init__()
        self.inplace = inplace

    def forward(self, inputs: Tensor) -> Tensor:
        if self.inplace:
            return inputs.mul_(inputs.sigmoid())
        else:
            return inputs * inputs.sigmoid()


class PointwiseConv1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 0,
                 ) -> None:
        super(PointwiseConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            groups=in_channels,
            padding=padding,
        )


    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 ) -> None:
        super(DepthwiseConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )


    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 32,
                 stride: int = 1,
                 padding: int = 15,
                 conv_expansionfactor: int=2,
                 conv_dropout: float=0.1
                 ) -> None:
        super(ConformerConvModule, self).__init__()
        self.layernorm = nn.LayerNorm(in_channels)
        self.pointwiseconv1d1 = PointwiseConv1D(
            in_channels=in_channels,
            out_channels=in_channels * conv_expansionfactor
        )
        self.glu = GLU(dim=1)
        self.deepwiseconv1d = DepthwiseConv1D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.swish = Swish()
        self.batchnorm = nn.BatchNorm1d(num_features=in_channels)
        self.pointwiseconv1d2 = PointwiseConv1D(
            in_channels=in_channels,
            out_channels=in_channels
        )
        self.dropout = nn.Dropout(conv_dropout)


    def forward(self, inputs: Tensor) -> Tensor:
        layernorm_output = self.layernorm(inputs)
        layernorm_output = layernorm_output.permute(0, 2, 1)
        pointwiseconv1d1_output = self.pointwiseconv1d1(layernorm_output)
        glu_output = self.glu(pointwiseconv1d1_output)
        deepwiseconv1d_output = self.deepwiseconv1d(glu_output)
        batchnorm_output = self.batchnorm(deepwiseconv1d_output)
        swish_output = self.swish(batchnorm_output)
        pointwiseconv1d2_output = self.pointwiseconv1d2(swish_output)
        pointwiseconv1d2_output = pointwiseconv1d2_output.permute(0, 2, 1)
        outputs = self.dropout(pointwiseconv1d2_output)

        return outputs


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self,
                 num_features: int,
                 max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, num_features, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * -(math.log(10000.0) / num_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 attn_dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        self.position_embedding = PositionalEncoding(self.num_features)
        self.layernorm = nn.LayerNorm(num_features)
        self.dim_heads = int(self.num_features/self.num_heads)
        self.scale = self.dim_heads ** -0.5
        self.to_queries = nn.Linear(in_features=num_features, out_features=num_features)
        self.to_keys = nn.Linear(in_features=num_features, out_features=num_features)
        self.to_values = nn.Linear(in_features=num_features, out_features=num_features)
        self.to_outputs = nn.Linear(in_features=self.dim_heads * self.num_heads, out_features=num_features)
        self.to_posembedding = nn.Linear(in_features=self.num_features, out_features=self.num_features, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.dim_heads))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.dim_heads))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        assert self.dim_heads * self.num_heads == self.num_features

    def forward(self,
                inputs: Tensor,
                ) -> Tensor:

        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.position_embedding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        inputs = self.layernorm(inputs)
        queries = self.to_queries(inputs).view(batch_size, -1, self.num_heads, self.dim_heads)
        keys = self.to_keys(inputs).view(batch_size, -1, self.num_heads, self.dim_heads).permute(0, 2, 1, 3)
        values = self.to_values(inputs).view(batch_size, -1, self.num_heads, self.dim_heads).permute(0, 2, 1, 3)
        pos_embedding = self.to_posembedding(pos_embedding).view(batch_size, -1, self.num_heads, self.dim_heads)
        #content_score = torch.matmul((queries + self.u_bias).transpose(1, 2), keys.transpose(2, 3))
        #pos_score = torch.matmul((queries + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        #pos_score = self._relative_shift(pos_score)
        content_score = torch.einsum('b h i d, b h j d -> b h i j', (queries+self.u_bias).permute(0, 2, 1, 3), keys)
        pos_score = torch.einsum('b h i d, b h j d -> b h i j', (queries + self.v_bias).permute(0, 2, 1, 3), pos_embedding.permute(0, 2, 1, 3))
        pos_score = self._relative_shift(pos_score)
        score = (content_score+pos_score) * self.scale
        attention = score.softmax(dim=-1)

        head = torch.einsum(' b h i j, b h j d -> b h i d', attention, values)

        h = head.size(1)
        i = head.size(2)
        d = head.size(3)
        concathead = head.view(batch_size, i, h*d)
        outputs = self.to_outputs(concathead)


        return outputs

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class FeedForwardModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        ff_expansionfactor: int = 2,
        ff_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features * ff_expansionfactor)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(ff_dropout)
        self.linear2 = nn.Linear(in_features=in_features * ff_expansionfactor, out_features=in_features)
        self.dropout2 = nn.Dropout(ff_dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        layernorm = self.layernorm(inputs)
        linear1 = self.linear1(layernorm)
        swish = self.swish(linear1)
        dropout1 = self.dropout1(swish)
        linear2 = self.linear2(dropout1)
        outputs = self.dropout2(linear2)

        return outputs


class ConformerBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        ff_expansionfactor: int = 2,
        ff_dropout: float = 0.1,
        kernel_size: int = 31,
        conv_expansionfactor: int = 2,
        conv_dropout: float = 0.1,
        num_heads: int=8,
        attn_dropout: float=0.1

    ) -> None:
        super().__init__()
        self.feedforward1 = FeedForwardModule(in_features=num_features,
                                              ff_expansionfactor=ff_expansionfactor,
                                              ff_dropout=ff_dropout)
        self.convolution = ConformerConvModule(in_channels=num_features,
                                               kernel_size=kernel_size,
                                               conv_expansionfactor=conv_expansionfactor,
                                               conv_dropout=conv_dropout)
        self.multiheadattention = MultiHeadAttention(num_features=num_features,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        self.feedforward2 = FeedForwardModule(in_features=num_features,
                                              ff_expansionfactor=ff_expansionfactor,
                                              ff_dropout=ff_dropout)


    def forward(self, inputs):
        outputs1 = self.feedforward1(inputs)
        outputs2 = 0.5 * outputs1+self.convolution(outputs1)
        outputs3 = outputs2+self.multiheadattention(outputs2)
        outputs = 0.5 * outputs3+self.feedforward2(outputs3)
        return outputs


