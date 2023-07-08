import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss,CrossEntropyLoss
from torch import optim
from torch.autograd import Variable
import os
from dataset import MyDataset, collate_fn
from source_model import Encoder, Decoder
from collections import Counter




gpu_id = 0
gpu_str = 'cuda:{}'.format(gpu_id)
device = torch.device(gpu_str if torch.cuda.is_available() else 'cpu')
print(device)

torch.backends.cudnn.benchmark = False

#path2 = 'F:\data\DNA_Micro_Disks\Traintestsplit'
path2 = 'E:\Traintestsplit'
ratio = 0.9
to_file_feature = os.path.join(path2, 'microdisks_train_feature_30_{}.txt'.format(ratio))
to_file_label = os.path.join(path2, 'microdisks_train_label_30_{}.txt'.format(ratio))

train_set = MyDataset(root_dir=to_file_feature, label_dir=to_file_label)


train_dl = DataLoader(dataset=train_set, collate_fn=collate_fn, batch_size=32, shuffle=True)

print('Finish loading')

def statistics(x,y):
    z=x.eq(y).int()
    acc=torch.sum(z, dim=1)
    num=torch.sum(acc)
    dim=x.size(0)

    return num/dim


Encoder=Encoder(in_channels=4,
                dim=64,
                ).to(device)

Decoder=Decoder(
    in_channels=5,
    lstm_hidden_dim=64,
    num_layers=1,
    rnn_dropout_p=0.1
).to(device)


criterion1 = CrossEntropyLoss()
encoder_optimizer = torch.optim.Adam(Encoder.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=False)
decoder_optimizer = torch.optim.Adam(Decoder.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=False)

#optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_optimizer, 0.95, last_epoch=-1)
decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, last_epoch=-1)

Encoder.train()
Decoder.train()


def train_loop(epoch, trainloader, Encoder, Decoder, criterion, encoder_optimizer, decoder_optimizer):
    sample = {}
    for t in range(epoch):
        loss0 = []
        size = len(trainloader.dataset)
        current=0
        for i, data in enumerate(trainloader):
            inputs, labels, decoder_input = data

            inputs, labels, decoder_input = Variable(inputs.float()).to(device), Variable(labels).to(device), Variable(decoder_input.float()).to(device)

            encoder_outputs, hidden = Encoder(inputs)

            l = decoder_input.size(1)
            if encoder_outputs.size(1)>l:
                encoder_outputs, _ = torch.split(encoder_outputs, l, dim=1)

            decoder_outputs, hidden = Decoder(decoder_input, encoder_outputs, hidden)



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

    torch.save(Encoder.state_dict(), 'encoder_train1_twod.pth')
    torch.save(Decoder.state_dict(), 'decoder_train1_twod.pth')
    return sample





name='train1'

epoch = 30
train_loss = train_loop(epoch, train_dl, Encoder, Decoder, criterion1, encoder_optimizer, decoder_optimizer)
torch.save(train_loss, 'loss_{}'.format(name))
print('Finished Training')

if __name__ == '__main__':
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    


