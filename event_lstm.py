import event

import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
torch.manual_seed(12345)


class Net(nn.Module):
    def __init__(self, len_voc, emb_size, hidden_size):
        super(Net, self).__init__()
        self.embs = nn.Embedding(len_voc, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.hlayer = nn.Linear(2 * emb_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.hlayer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.top = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, events, timexs, seqs):
        event = self.embs(events)
        time = self.embs(timexs)
        seq = self.embs(seqs)
        seq = self.gru(seq)
        output = torch.cat((event, time, seq[-1].view(1, -1)), 1)
        output = self.relu(self.hlayer(output))
        output = self.relu2(self.hlayer2(output))
        output = self.sigmoid(self.top(output))
        return output

#    def init_h0(self, N):
#        return Variable(torch.randn(1, N, self.hidden_size))


def train(net, dataX, dataY, vocab, nepochs, batch_size):
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
            batchS = batchS.transpose(0, 1)
            batchY = dataY[b*batch_size:b*batch_size+batch_size]
            batchW = [0.99 if y == 1 else 0.01 for y in batchY]
            batchW = Variable(torch.FloatTensor([batchW]))
            batchY = Variable(torch.FloatTensor([batchY]))

            # Clear gradients
            net.zero_grad()

            # Forward pass
            y_pred = net(batchE, batchT, batchS)

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
        batchS = batchS.transpose(0, 1)

        # Forward pass
        y_pred = net(batchE, batchT, batchS)
        prediction.append(y_pred.data[0][0])

        sys.stdout.write('\r[batch: %3d/%3d]' % (b + 1, batches))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return prediction


cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train_corenlp/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train/TimeBank/'
train_x, train_y, vocab = event.get_data(cnlp, anafora)
cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_corenlp/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_gold/'
test_x, test_y = event.get_test_data(cnlp, anafora, vocab)

net = Net(len(vocab), 200, 100)
train(net, train_x, train_y, vocab, 3, 1)
prediction = predict(net, test_x, 1)
print(prediction)
predicition = np.array(list(map(round, prediction)))
test_y = np.array(test_y)

print('Pred: %d - True: %d - Acc: %d' % (np.sum(predicition == 1), np.sum(test_y == 1), np.sum(predicition + test_y == 2)))
