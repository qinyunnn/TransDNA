import os
import sys
sys.path.append('../')


import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss,CrossEntropyLoss
from torch import optim
from torch.autograd import Variable
import argparse
from dataset import MyDataset, CustomSampler, CustomBatchSampler, collater
from source_model import Encoder, Decoder
from collections import Counter
from utils import statistics


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_data',
                        required=True,
                        help='Train data file',
                        default='./examples/data/source_data/reads.txt')
    parser.add_argument('--ground_truth',
                        required=True,
                        help='Label file',
                        default='./examples/data/source_data/reference.txt')
    parser.add_argument('--padding_length',
                        required=True,
                        type=int,
                        help='Maximum length of training data')
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


    encoder=Encoder(
        in_channels=4,
        dim=args.dim).to(device)

    decoder=Decoder(
        in_channels=5,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_layers=args.num_layers,
        rnn_dropout_p=args.rnn_dropout_p).to(device)


    criterion = CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=False)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=False)

    encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_optimizer, 0.95, last_epoch=-1)
    decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, last_epoch=-1)

    encoder.train()
    decoder.train()





    def train_loop(epoch, trainloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer):
        sample = {}
        for t in range(epoch):
            loss0 = []
            size = len(trainloader.dataset)
            current=0
            for i, data in enumerate(trainloader):
                inputs, labels, decoder_input = data

                inputs, labels, decoder_input = Variable(inputs.float()).to(device), Variable(labels).to(device), Variable(decoder_input.float()).to(device)

                encoder_outputs, hidden = encoder(inputs)

                l = decoder_input.size(1)
                if encoder_outputs.size(1)>l:
                    encoder_outputs, _ = torch.split(encoder_outputs, l, dim=1)

                decoder_outputs, hidden = decoder(decoder_input, encoder_outputs, hidden)



                outputs = decoder_outputs.permute(0, 2, 1)
                loss = criterion(outputs, labels)
                y = outputs.argmax(dim=1)

                z = statistics(y, labels)

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.requires_grad_(True)

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                loss_fn = loss.item() / len(outputs)
                current = current+len(outputs)
                if i % 10 == 0:
                    print(f'Epoch {t + 1} \n ----------------------\nloss: {loss_fn:>7f}  [{current:>5d}/{size:>5d}]')
                    print(z)
                loss0.append(loss_fn)
                sample.setdefault(t + 1, []).append(loss0)
                encoder_scheduler.step()
                decoder_scheduler.step()

            torch.save(encoder.state_dict(), os.path.join(args.model_dir,'encoder_para_{}.pth'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(args.model_dir,'decoder_para_{}.pth'.format(epoch)))
        return sample




    epoch = args.epoch
    train_loss = train_loop(epoch, train_dl, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
    torch.save(train_loss, os.path.join(args.model_dir, 'loss_{}.pth'.format(args.epoch)))
    print('Finished Training')



if __name__ == '__main__':
    main()