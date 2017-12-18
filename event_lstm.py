import event

import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
np.random.seed(12345)
torch.manual_seed(12345)


class Net(nn.Module):
    def __init__(self, len_voc, len_dinv, wemb_size, demb_size, hidden_size):
        super(Net, self).__init__()
        self.gru_layers = 1
        self.tok_embs = nn.Embedding(len_voc, wemb_size)
        self.hidden_size = hidden_size
        self.sent_gru = nn.GRU(wemb_size, hidden_size, bidirectional=True, num_layers=self.gru_layers)
        self.dep_embs = nn.Embedding(len_dinv, demb_size)
        self.dep_gru = nn.GRU(demb_size, hidden_size, bidirectional=True, num_layers=self.gru_layers)
        self.hlayer = nn.Linear(2 * wemb_size + 2 * hidden_size + 2 * hidden_size, hidden_size)
        #self.hlayer = nn.Linear(2 * wemb_size + 2 * demb_size + 2 * hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.hlayer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.top = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, events, timexs, seqs, path):
        event = self.tok_embs(events)
        time = self.tok_embs(timexs)
        seq = self.tok_embs(seqs)
        seq, _ = self.sent_gru(seq, self.s_h0)
        dseq = self.dep_embs(path)
        dseq, _ = self.dep_gru(dseq, self.d_h0)
        output = torch.cat((event, time, seq[-1].view(1, -1), dseq[-1].view(1, -1)), 1)
        output = self.relu(self.hlayer(output))
        output = self.relu2(self.hlayer2(output))
        output = self.sigmoid(self.top(output))
        return output

    # def forward(self, events, timexs, seqs, path):
    #     event = self.tok_embs(events)
    #     time = self.tok_embs(timexs)
    #     seq = self.tok_embs(seqs)
    #     seq, _ = self.sent_gru(seq, self.s_h0)
    #     d1 = self.dep_embs(path[0])
    #     d2 = self.dep_embs(path[-1])
    #     output = torch.cat((event, time, seq[-1].view(1, -1), d1, d2), 1)
    #     output = self.relu(self.hlayer(output))
    #     output = self.relu2(self.hlayer2(output))
    #     output = self.sigmoid(self.top(output))
    #     return output
    
    def init_h0(self, N):
        self.s_h0 = Variable(torch.randn(self.gru_layers * 2, N, self.hidden_size))
        self.d_h0 = Variable(torch.randn(self.gru_layers * 2, N, self.hidden_size))


def train(net, dataX, dataY, vocab, nepochs, batch_size, class_weight):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters)
    batches = math.ceil(len(dataX) / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            batchX = dataX[b*batch_size:b*batch_size+batch_size]
            batchE = Variable(torch.LongTensor([batchX[0][0]]))
            batchT = Variable(torch.LongTensor([batchX[0][1]]))
            batchS = Variable(torch.LongTensor([batchX[0][2]]))
            batchP = Variable(torch.LongTensor([batchX[0][3]]))
            batchS = batchS.transpose(0, 1)
            batchP = batchP.transpose(0, 1)
            batchY = dataY[b*batch_size:b*batch_size+batch_size]
            batchW = [class_weight if y == 1 else 1. - class_weight for y in batchY]
            batchW = Variable(torch.FloatTensor([batchW]))
            batchY = Variable(torch.FloatTensor([batchY]))

            # Clear gradients
            net.zero_grad()

            # Forward pass
            net.init_h0(batchE.size(0))
            y_pred = net(batchE, batchT, batchS, batchP)

            # Compute loss
            criterion = nn.BCELoss(weight=batchW)
            loss = criterion(y_pred, batchY)

            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d/%3d] loss: %.3f' % (e + 1, b + 1, batches, running_loss / (b+1)))
            sys.stdout.flush()

            # Backward propagation and update the weights.
            loss.backward()
            optimizer.step()
        sys.stdout.write('\n')
        sys.stdout.flush()


def predict(net, dataX, batch_size):
    prediction = list()
    batches = math.ceil(len(dataX) / batch_size)
    for b in range(batches):
        # Get minibatch samples
        batchX = dataX[b*batch_size:b*batch_size+batch_size]
        batchE = Variable(torch.LongTensor([batchX[0][0]]))
        batchT = Variable(torch.LongTensor([batchX[0][1]]))
        batchS = Variable(torch.LongTensor([batchX[0][2]]))
        batchP = Variable(torch.LongTensor([batchX[0][3]]))
        batchS = batchS.transpose(0, 1)
        batchP = batchP.transpose(0, 1)

        # Forward pass
        net.init_h0(batchE.size(0))
        y_pred = net(batchE, batchT, batchS, batchP)
        prediction.append(y_pred.data[0][0])

        sys.stdout.write('\r[batch: %3d/%3d]' % (b + 1, batches))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return prediction


embfile = 'embs.list'
vocab, embs = event.get_embs(embfile)
cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train_corenlp/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train/train/'
# train_x, train_y, vocab, dep_inventory = event.get_data(cnlp, anafora)
train_x, train_y, dep_inventory = event.get_vocab_data(cnlp, anafora, vocab)

# atrain_y = np.array(train_y)
# posidx = list(np.where(atrain_y == 1)[0])
# negidx = list(np.random.choice(np.where(atrain_y == 0)[0], 200, replace=False))
# nx = list()
# ny = list()
# for e, (x, y) in enumerate(zip(train_x, train_y)):
#     if e in posidx or e in negidx:
#         nx.append(x)
#         ny.append(y)
# train_x = nx
# train_y = ny

# train_x, train_y = event.shuffle(train_x, train_y)
#cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_corenlp/'
#anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_gold/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train/dev/'
test_x, test_y, _ = event.get_vocab_data(cnlp, anafora, vocab, dep_inv=dep_inventory)

net = Net(len(vocab), len(dep_inventory), 200, 100, 100)
# net.tok_embs.weight.data.copy_(torch.from_numpy(np.array(embs)))
print(net)
pos_percent = np.sum(train_y)/len(train_y)
train(net, train_x, train_y, vocab, 10, 1, 0.5)
prediction = predict(net, test_x, 1)
# print(prediction)
predicition = np.array(list(map(round, prediction)))
test_y = np.array(test_y)

print('Pred: %d - True: %d - Acc: %d' % (np.sum(predicition == 1), np.sum(test_y == 1), np.sum(predicition + test_y == 2)))
