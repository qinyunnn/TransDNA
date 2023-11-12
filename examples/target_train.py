import os
import sys
sys.path.append('../')


import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.autograd import Variable
import argparse
from dataset import MyDataset, CustomSampler, CustomBatchSampler, collater
from transfer_model import Encoder, Decoder, Domain
from collections import Counter
from utils import statistics


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_data',
                        required=True,
                        help='Train data file',
                        default='./examples/data/reads.txt')
    parser.add_argument('--ground_truth',
                        required=True,
                        help='Label file',
                        default='./examples/data/reference.txt')
    parser.add_argument('--source_padding_length',
                        required=True,
                        type=int,
                        help='Maximum length of source data')
    parser.add_argument('--target_padding_length',
                        required=True,
                        type=int,
                        help='Maximum length of target data')
    parser.add_argument('--pretrained_encoder',
                        required=True,
                        help='Pretrained encoder')
    arser.add_argument('--pretrained_decoder',
                       required=True,
                       help='Pretrained decoder')
    parser.add_argument('--model_dir',
                        required=True,
                        help='Save model dir')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size',
                        default='32')
    parser.add_argument('--epoch',
                        required=True,
                        type=int,
                        help='epoch',
                        default='10')
    parser.add_argument('--dim',
                        default=64,
                        type=int,
                        help='Feature dim')
    parser.add_argument('--num_layers',
                        default=1,
                        type=int,
                        help='Number of Conformer block')
    parser.add_argument('--lstm_hidden_dim',
                        default=64,
                        type=int,
                        help='Lstm hidden dim')
    parser.add_argument('--rnn_dropout_p',
                        default=0.1,
                        help='Dropout of RNN block')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gpu_id = 0
    gpu_str = 'cuda:{}'.format(gpu_id)
    device = torch.device(gpu_str if torch.cuda.is_available() else 'cpu')
    print(device)

    collate_fn = collater(args.padding_length)
    train_set=MyDataset(root_dir=args.train_data,
                        label_dir=args.ground_truth)

    train_cs=CustomSampler(data=train_set)
    train_bs=CustomBatchSampler(sampler=train_cs, batch_size=args.batch_size, drop_last=False)
    train_dl=DataLoader(dataset=train_set, batch_sampler=train_bs, collate_fn=collate_fn)
    print('Finish loading')


    Source_Encoder=Encoder(
        in_channels=4,
        dim=args.dim).to(device)

    Source_Decoder=Decoder(
        in_channels=5,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_layers=args.num_layers,
        rnn_dropout_p=args.rnn_dropout_p).to(device)

    domain = Domain(source_length=args.source_padding_length,
                    target_length=args.target_padding_length).to(device)

    Target_Decoder = Decoder(
        in_channels=5,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_layers=args.num_layers,
        rnn_dropout_p=args.rnn_dropout_p).to(device)

    pretrained_encoder_dict = torch.load(args.pretrained_encoder, map_location='cuda:0')
    Source_Encoder.load_state_dict(pretrained_encoder_dict)

    pretrained_decoder_dict = torch.load(args.pretrained_decoder, map_location='cuda:0')
    Source_Decoder.load_state_dict(pretrained_decoder_dict)


    criterion = CrossEntropyLoss()
    source_encoder_optimizer = torch.optim.Adam(Source_Encoder.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9,
                                                weight_decay=1e-4, amsgrad=False)
    source_decoder_optimizer = torch.optim.Adam(Source_Decoder.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9,
                                                weight_decay=1e-4, amsgrad=False)
    target_decoder_optimizer = torch.optim.Adam(Target_Decoder.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9,
                                                weight_decay=1e-4, amsgrad=False)
    domain_optimizer = torch.optim.Adam(domain.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4,
                                        amsgrad=False)

    source_encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(source_encoder_optimizer, 0.95, last_epoch=-1)
    source_decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(source_decoder_optimizer, 0.95, last_epoch=-1)
    target_decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(target_decoder_optimizer, 0.95, last_epoch=-1)
    domain_scheduler = torch.optim.lr_scheduler.ExponentialLR(domain_optimizer, 0.95, last_epoch=-1)

    def train_loop(epoch, S_train_dl, T_train_dl, Source_Encoder, Source_Decoder, domain, Target_Decoder, criterion,
                   source_encoder_optimizer,
                   source_decoder_optimizer, target_decoder_optimizer, domain_optimizer):
        sample = {}

        for t in range(epoch):
            len_dataloader = min(len(S_train_dl), len(T_train_dl))
            data_source_iter = iter(Source_train_dl)
            data_target_iter = iter(Target_train_dl)
            loss0 = []
            size = len(Target_train_dl.dataset)
            iter_num = 0
            current = 0
            while iter_num < len_dataloader:
                data_source = data_source_iter.next()
                S_inputs, S_labels, S_decoder_input = data_source
                S_inputs, S_labels, S_decoder_input = Variable(S_inputs.float()).to(device), Variable(S_labels).to(
                    device), Variable(S_decoder_input.float()).to(device)

                data_target = data_target_iter.next()
                T_inputs, T_labels, T_decoder_input = data_target
                T_inputs, T_labels, T_decoder_input = Variable(T_inputs.float()).to(device), Variable(T_labels).to(
                    device), Variable(T_decoder_input.float()).to(device)

                S_encoder_outputs, S_hidden = Source_Encoder(S_inputs)
                S_encoder_outputs_2 = S_encoder_outputs[:]
                S_l = S_decoder_input.size(1)
                if S_encoder_outputs.size(1) > S_l:
                    S_encoder_outputs, _ = torch.split(S_encoder_outputs, S_l, dim=1)

                # S_decoder_outputs, hidden = Source_Decoder(torch.cat([S_decoder_input, S_encoder_outputs], dim=2), S_encoder_outputs, S_hidden)
                S_decoder_outputs, hidden = Source_Decoder(S_decoder_input, S_encoder_outputs, S_hidden)

                S_decoder_outputs = S_decoder_outputs.permute(0, 2, 1)

                T_encoder_outputs, T_hidden = Source_Encoder(T_inputs)
                T_encoder_outputs_2 = T_encoder_outputs[:]
                T_l = T_decoder_input.size(1)
                if T_encoder_outputs.size(1) > T_l:
                    T_encoder_outputs, _ = torch.split(T_encoder_outputs, T_l, dim=1)

                T_decoder_outputs, hidden = Target_Decoder(T_decoder_input, T_encoder_outputs, T_hidden)
                T_decoder_outputs = T_decoder_outputs.permute(0, 2, 1)

                source_domain_output, target_domain_output = domain(S_encoder_outputs_2, T_encoder_outputs_2)

                loss_decoder_source = criterion(S_decoder_outputs, S_labels)
                y_source = S_decoder_outputs.argmax(dim=1)

                loss_decoder_target = criterion(T_decoder_outputs, T_labels)
                y_target = T_decoder_outputs.argmax(dim=1)

                z_source = statistics(y_source, S_labels)
                z_target = statistics(y_target, T_labels)

                loss_domain = mmd_loss(source_domain_output, target_domain_output)
                loss = loss_decoder_source + loss_decoder_target + 0.5 * loss_domain
                print(loss_decoder_source)
                print(loss_decoder_target)
                print(loss_domain)

                source_encoder_optimizer.zero_grad()
                source_decoder_optimizer.zero_grad()
                target_decoder_optimizer.zero_grad()
                domain_optimizer.zero_grad()
                loss.requires_grad_(True)

                loss.backward()
                source_encoder_optimizer.step()
                source_decoder_optimizer.step()
                target_decoder_optimizer.step()
                domain_optimizer.step()

                loss_fn = loss.item() / len(T_decoder_outputs)
                current = current + len(T_decoder_outputs)

                print(f'Epoch {t + 1} \n ----------------------\nloss: {loss_fn:>7f}  [{current:>5d}/{size:>5d}]')
                print(loss_decoder_target.item() / len(T_decoder_outputs))
                print(z_source)
                print(z_target)
                loss0.append(loss_fn)
                sample.setdefault(t + 1, []).append(loss0)
                iter_num += 1
                source_encoder_scheduler.step()
                source_decoder_scheduler.step()
                target_decoder_scheduler.step()
                domain_scheduler.step()

            torch.save(Source_Encoder.state_dict(), os.path.join(args.model_dir, 'transfer_encoder_{}.pth'.format(epoch)))
            torch.save(Target_Decoder.state_dict(), os.path.join(args.model_dir, 'transfer_decoder_{}.pth'.format(epoch)))

        return sample




    epoch = args.epoch
    train_loss = train_loop(epoch, Source_train_dl, Target_train_dl, Source_Encoder, Source_Decoder, domain,
                            Target_Decoder, criterion1, source_encoder_optimizer,
                            source_decoder_optimizer, target_decoder_optimizer, domain_optimizer)
    torch.save(train_loss, os.path.join(args.model_dir, 'loss_{}.pth'.format(args.epoch)))
    print('Finished Training')



if __name__ == '__main__':
    main()