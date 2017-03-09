#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:11:06 2017

@author: egoitz
"""
import sys
from seq2seq.models import SimpleSeq2Seq, Seq2Seq
import numpy as np
np.random.seed(12345)
from keras.utils.test_utils import keras_test

from keras.layers import Input, LSTM, GRU, TimeDistributed, Dense, Masking, Dropout, Flatten
from keras.models import Model, Sequential


import getseqs

input_length = 10
input_dim = 2
output_length = 8
output_dim = 3
samples = 100

# LSTM Parameters
epochs = 10 # Number of epochs to cycle through data
batch_size = 100 # Train on this many examples at once
learning_rate = 0.001 # Learning rate
val_split = 0.25


train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'

links, entities, sequences,  max_seq = getseqs.getdata(train_path)
(entitylists, transitions, newlinks, 
 max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)
max_seq = 10
(types, types2idx,
 parentsTypes, parentsTypes2idx,
 _,_) = getseqs.build_vocabs(links, entities, sequences)
data_x, data_y, out_class = getseqs.data_to_seq2seq_single_vector(links, entities, sequences, max_seq,
                                                                  transitions, max_trans,
                                                                  types2idx, len(types),
                                                                  parentsTypes2idx, len(parentsTypes),
                                                                  trans2idx, len(transOp))
feat_size = len(types) + len(parentsTypes)

#for seq in data_y:
#    for prediction in seq:
#        print prediction
#        p = np.argmax(prediction)
#        print p,transOp[p]
#    print ""
#sys.exit()


input_layer = Input(shape=(max_seq,feat_size), dtype='float32')
mask = (Masking(mask_value=0))(input_layer)
s2s = Seq2Seq(output_dim=len(transOp),
                      output_length=max_trans, 
                      input_shape=(max_seq, feat_size),
                      depth=2)(mask)

model = Model(input=input_layer, output=s2s)

#input_layer = Input(shape=(max_seq,feat_size), dtype='float32', name="inputs")
#masked_input = Masking(mask_value=-1)(input_layer)
#enco = GRU(100, name="enco")(input_layer)
##deco = GRU(100, name="deco",return_sequences=True)(enco)
#top = TimeDistributed(Dense(len(transOp), activation='softmax', name="top"))(enco)
#model = Model(input=input_layer, output=top)
#model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])
#model.fit(data_x, data_y, batch_size=batch_size, nb_epoch=epochs, validation_split=val_split)

 
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.fit(data_x, data_y, nb_epoch=10)


entities, sequences,  _ = getseqs.get_testdata(test_path)
links = dict()
data_x, _, _ = getseqs.data_to_seq2seq_single_vector(links, entities, sequences, max_seq,
                                                     transitions, max_trans,
                                                     types2idx, len(types),
                                                     parentsTypes2idx, len(parentsTypes),
                                                     trans2idx, len(transOp))

predictions = model.predict(data_x)
for seq in predictions:
    for prediction in seq:
        p = np.argmax(prediction)
        #print prediction
        print p,transOp[p]
    print ""
