#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:37:22 2017

@author: egoitz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:00:24 2017

@author: egoitz
"""
import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
torch.manual_seed(12345)

import getseqs



class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hlayer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.hlayer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.hlayer3 = nn.Linear(hidden_size, 10)
        self.relu3 = nn.ReLU()
        self.top = nn.Linear(10, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        output = self.relu1(self.hlayer1(input))
        output = self.relu2(self.hlayer2(output))
        output = self.relu3(self.hlayer3(output))
        output = self.softmax(self.top(output))
        return output

def train(net, x, y, nepochs, batch_size):
    criterion = nn.NLLLoss()
    optimizer = optim.Adadelta(net.parameters())
    batches = math.ceil(x.size()[0] / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            bx = x[b*batch_size:b*batch_size+batch_size]
            by = y[b*batch_size:b*batch_size+batch_size]

            # Forward pass
            y_pred = net(bx)

            # Compute loss
            loss = criterion(y_pred, by)

            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d] loss: %.3f' % (e + 1, b + 1, running_loss / (b+1)))
            sys.stdout.flush()

            # Zero gradients, backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sys.stdout.write('\n')
        sys.stdout.flush()

# LSTM Parameters
epochs = 10 # Number of epochs to cycle through data
batch_size = 100 # Train on this many examples at once
learning_rate = 0.001 # Learning rate
val_split = 0.25

# Get and process data
# train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
# test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'

links, entities, sequences,  max_seq = getseqs.getdata(train_path)
max_seq = 10
(types, types2idx,
 parentsTypes, parentsTypes2dx,
 linkTypes, linkTypes2idx) = getseqs.build_vocabs(links, entities, sequences)
data_x, data_y, out_class = getseqs.data_to_pair2lab_vector(links, entities, sequences, max_seq,
                                                           types2idx, len(types),
                                                           parentsTypes2dx, len(parentsTypes),
                                                           linkTypes2idx, len(linkTypes))
feat_size = 2 * len(types) + 2 * len(parentsTypes) + 1

print ('Train x shape:', np.shape(data_x))
print ('Test y shape:', np.shape(data_y))


# The model
net = Net(feat_size, 200, len(linkTypes))
data_x = Variable(torch.from_numpy(data_x).float())
data_y = np.argmax(data_y,axis=1)
data_y = Variable(torch.from_numpy(data_y).long())
train(net, data_x, data_y, epochs, batch_size)

# Testing
entities, sequences,  _ = getseqs.get_testdata(test_path)
links = dict()
data_x, _, _ = getseqs.data_to_pair2lab_vector(links, entities, sequences, max_seq,
                                              types2idx, len(types),
                                              parentsTypes2dx, len(parentsTypes),
                                              linkTypes2idx, len(linkTypes))


print ('\nTest x shape:', np.shape(data_x))

data_x = Variable(torch.from_numpy(data_x).float())
predictions = net(data_x)
predictions = predictions.data.numpy()

labels = list()
for p in predictions:
    i = np.argmax(p)
    labels.append((linkTypes[i],0))

outputs = dict()
l = 0
for key in sequences.keys():
    for target in sorted(sequences[key]):
        xmlfile = entities[target][3]
        if xmlfile not in outputs:
            outputs[xmlfile] = dict()
        targetid = entities[target][4]
        if targetid not in outputs[xmlfile]:
            outputs[xmlfile][targetid] = dict()
        for entity in sorted(sequences[key]):
            if target != entity:
                if labels[l][0] != "O":
                    outputs[xmlfile][targetid][entities[entity][4]] = labels[l]
                l += 1

getseqs.print_outputs(test_path,out_path, outputs)

print ("")
