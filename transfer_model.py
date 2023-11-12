import torch
from torch import nn
from torch import Tensor
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
                 dropput_conv=0.1,
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
        self.dropout = nn.Dropout(p=dropput_conv)

    def forward(self, input: Tensor) -> Tensor:

        input = torch.transpose(input, 1, 2)
        output1 = self.sequential1(input)
        output3 = self.sequential3(input)
        output5 = self.sequential5(input)
        output7 = self.sequential7(input)
        output = torch.cat([output1, output3, output5, output7], dim=1)
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)

        return output #[batch_size, L, 128]



class Domain_specific(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels = 1,
                 conv_dropout_p=0.1,
                 ) -> None:
        super(Domain_specific, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels//8, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(p=conv_dropout_p)

    def forward(self, input: Tensor) -> Tensor:
        output = self.sequential(input)
        output = self.dropout(output)

        return output.squeeze(1)


class Domain(nn.Module):
    def __init__(self,
                 source_length: int,
                 target_length: int,
                 ) -> None:
        super(Domain, self).__init__()
        self.domainspecific1 = Domain_specific(in_channels=source_length)
        self.domainspecific3 = Domain_specific(in_channels=target_length)

    def forward(self, source_input, target_input) -> Tensor:
        source_output = self.domainspecific1(source_input)
        target_output = self.domainspecific3(target_input)
        return source_output, target_output



class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int,
                 ):
        super(Encoder, self).__init__()
        self.Conv = Conv(in_channels=in_channels, out_channels=dim)
        self.encoder = ConformerBlock(num_features=dim)

    def forward(self, input: Tensor) -> Tensor:
        output1 = self.Conv(input)
        output = self.encoder(output1)
        #output2 = output.permute(0, 2, 1)
        h0 = torch.sum(output, dim=1, keepdim=True)
        c0 = torch.sum(output, dim=1, keepdim=True)
        h0 = h0.permute(1, 0, 2)
        c0 = c0.permute(1, 0, 2)
        #h0 = self.project1(output2).permute(2, 0, 1)
        #c0 = self.project2(output2).permute(2, 0, 1)
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

        self.dropout = nn.Dropout(rnn_dropout_p)

        # encoder_hidden_states:[num_layers, batch_size, project_size]
        # input:[batch_size, L, 5]
    def forward(self, input: Tensor, encoder_output: Tensor, hidden: Tensor) -> Tensor:


        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)

        linear_input = torch.cat([encoder_output, output], dim=2)
        output = self.linear(linear_input)


        return output, hidden# [B, L, 4]









class TransferModel(nn.Module):
    def __init__(self,
                 source_noise_length: int,
                 target_noise_length: int,
                 in_channels: int,
                 dim: int,
                 num_layers=1,
                 ) -> None:
        super(TransferModel, self).__init__()
        self.Conv = Conv(in_channels=4)
        self.encoder = ConformerBlock(dim=dim,
                                      dim_head=16,
                                      heads=8,
                                      ff_mult=4,
                                      conv_expansion_factor=2,
                                      conv_kernel_size=31,
                                      attn_dropout=0.1,
                                      ff_dropout=0.1,
                                      conv_dropout=0.1
                                      )
        self.source_domainspecific = Domain_specific(
            in_channels = source_noise_length
        )
        self.target_domainspecific = Domain_specific(
            in_channels = target_noise_length
        )

        self.source_decoder = RNNBlock(
            in_length=source_noise_length,
            in_channels=in_channels,
            lstm_hidden_dim=dim,
            num_layers=num_layers,
        )
        self.target_decoder = RNNBlock(
            in_length=target_noise_length,
            in_channels=in_channels,
            lstm_hidden_dim=dim,
            num_layers=num_layers
        )

    def forward(self,
                source_input: Tensor,
                target_input: Tensor,
                source_decoder_input: Tensor,
                target_decoder_input: Tensor
                ) -> Tensor:
        source_input_embedding = self.Conv(source_input)
        target_input_embedding = self.Conv(target_input)
        source_encoder_output = self.encoder(source_input_embedding)
        target_encoder_output = self.encoder(target_input_embedding)
        source_domain_output = self.source_domainspecific(source_encoder_output)
        target_domain_output = self.target_domainspecific(target_encoder_output)
        source_output = self.source_decoder(source_decoder_input, source_encoder_output)
        target_output = self.target_decoder(target_decoder_input, target_encoder_output)
        return source_domain_output, target_domain_output, source_output, target_output


