#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:11:06 2017

@author: egoitz
"""
import sys
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
np.random.seed(12345)
from keras.optimizers import Adam
from keras.utils.test_utils import keras_test

from keras.layers import Input, LSTM, GRU, TimeDistributed, Dense, Masking, Dropout, Flatten, Embedding
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical


import getseqs

input_length = 10
input_dim = 2
output_length = 8
output_dim = 3
samples = 100

# LSTM Parameters
epochs = 500 # Number of epochs to cycle through data
batch_size = 100 # Train on this many examples at once
learning_rate = 0.001 # Learning rate
val_split = 0.25

train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'

links, entities, sequences,  max_seq = getseqs.getdata(train_path)
(entitylists, transitions, newlinks, 
 max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)
max_seq = 10
(types, types2idx,
 parentsTypes, parentsTypes2idx,
 _, _) = getseqs.build_vocabs(links, entities, sequences)
# data_x, data_y, out_class = getseqs.data_to_seq2seq_single_vector(links, entities, sequences, max_seq,
#                                                                   transitions, max_trans,
#                                                                   types2idx, len(types),
#                                                                   parentsTypes2idx, len(parentsTypes),
#                                                                   trans2idx, len(transOp))
data_x, data_y, out_class = getseqs.data_to_seq2seq(links, entities, sequences, max_seq,
                                                                  transitions, max_trans,
                                                                  types2idx, len(types),
                                                                  parentsTypes2idx, len(parentsTypes),
                                                                  trans2idx, len(transOp))
feat_size = len(types)


model = Sequential()
model.add(Embedding(feat_size, 128, input_length=max_seq))
model.add(Dropout(0.3))
model.add(AttentionSeq2Seq(output_dim=128,
                      output_length=max_trans,
                      input_length=max_seq,
                         input_dim=128,
                         hidden_dim=128,
                         depth=1,
                         bidirectional=True,
                         dropout=0.2
                         ))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(TimeDistributed(Dense(len(transOp), activation='softmax')))
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
print (np.shape(data_x))
print (np.shape(data_y))
model.fit(data_x, data_y, nb_epoch=epochs)


entities, sequences,  _ = getseqs.get_testdata(test_path)
links = dict()
# data_x, _, _ = getseqs.data_to_seq2seq_single_vector(links, entities, sequences, max_seq,
#                                                      transitions, max_trans,
#                                                      types2idx, len(types),
#                                                      parentsTypes2idx, len(parentsTypes),
#                                                      trans2idx, len(transOp))
data_x, _, _ = getseqs.data_to_seq2seq(links, entities, sequences, max_seq,
                                                                  transitions, max_trans,
                                                                  types2idx, len(types),
                                                                  parentsTypes2idx, len(parentsTypes),
                                                                  trans2idx, len(transOp))

predictions = model.predict(data_x)

transitions = list()
for p in predictions:
    seq = list()
    for s in p:
        i = np.argmax(s)
        seq.append(transOp[i])
    transitions.append(seq)

entitylists = list()
for s in sequences:
    entitylist = list()
    for e in sequences[s]:
        begin = entities[e][0].split(',')[0]
        entitylist.append((e, int(begin)))
    #entitylist = sorted(entitylist, key=lambda x: x[1])
    entitylists.append(entitylist)

outputs = getseqs.build_graph(entitylists, entities, transitions)
getseqs.print_outputs(test_path, out_path, outputs)
