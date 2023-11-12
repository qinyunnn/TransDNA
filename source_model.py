import torch
from torch import nn
from torch import Tensor, einsum
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple
from ConformerBlock import ConformerBlock





class Conv(nn.Module):
    def __init__(self,
                 in_channels=4,
                 out_channels=64,
                 conv_dropout=0.1,
                 ) -> None:
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.sequential1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.sequential3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sequential5 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.sequential7 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=conv_dropout)

    def forward(self, input: Tensor) -> Tensor:

        input = torch.transpose(input, 1, 2)
        output1 = self.sequential1(input)
        output3 = self.sequential3(input)
        output5 = self.sequential5(input)
        output7 = self.sequential7(input)
        output = torch.cat([output1, output3, output5, output7], dim=1)
        output = self.dropout(output)
        output = torch.transpose(output, 1, 2)

        return output #[batch_size, L, 128]


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int
                 ):
        super(Encoder, self).__init__()
        self.Conv = Conv(in_channels=in_channels, out_channels=dim)
        self.encoder = ConformerBlock(num_features=dim)


    def forward(self, input: Tensor) -> Tensor:
        output1 = self.Conv(input)
        output = self.encoder(output1)
        h0 = torch.sum(output, dim=1, keepdim=True)
        c0 = torch.sum(output, dim=1, keepdim=True)

        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = c0.permute(1, 0, 2).contiguous()

        return output, (h0, c0)



class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 lstm_hidden_dim: int,
                 num_layers: int,
                 rnn_dropout_p=0.1,
                 ) -> None:
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_dim, num_layers=num_layers, bidirectional=False,
                           batch_first=True)

        self.linear = nn.Linear(in_features=lstm_hidden_dim*2, out_features=5)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rnn_dropout_p)

        # encoder_hidden_states:[num_layers, batch_size, project_size]
        # input:[batch_size, L, 5]
    def forward(self, input: Tensor, encoder_output: Tensor, hidden: Tensor) -> Tensor:

        output, hidden = self.rnn(input, hidden)

        #a = torch.einsum('b m i, b n i -> b m n', encoder_output, output) * self.scale
        #b = a.softmax(dim=-1)
        #c = torch.einsum('b m n, b m i -> b n i', b, encoder_output)

        linear_input = torch.cat([encoder_output, output], dim=2)
        output1 = self.linear(linear_input)
        output1 = self.dropout(output1)

        return output1, hidden






