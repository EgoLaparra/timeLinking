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
import numpy as np
np.random.seed(55555)
from keras.layers import Input, LSTM, GRU, TimeDistributed, Dense, Masking, Dropout, Bidirectional, Lambda, Flatten
from keras.models import Model
from keras.layers.core import K

import getseqs

def Adjancency(x):
    x_shape = K.shape(x)
    m = K.tile(x, x_shape[0])
    #m = K.reshape(m, (x_shape[0], x_shape[0], x_shape[1]))
    #m_T = K.permute_dimensions(m,(1,0,2))
    #return m * m_T
    return m
    
# LSTM Parameters
epochs = 10 # Number of epochs to cycle through data
batch_size = 100 # Train on this many examples at once
learning_rate = 0.001 # Learning rate
val_split = 0.25

# Get and process data
train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'

links, entities, sequences,  max_seq = getseqs.getdata(train_path)
max_seq = 10
(types, types2idx,
 parentsTypes, parentsTypes2dx,
 linkTypes, linkTypes2idx) = getseqs.build_vocabs(links, entities, sequences)
data_x, data_y, out_class = getseqs.data_to_seq2graph(links, entities, sequences, max_seq,
                                                           types2idx, len(types),
                                                           parentsTypes2dx, len(parentsTypes),
                                                           linkTypes2idx, len(linkTypes))
feat_size = len(types) + len(parentsTypes)

print (np.shape(data_x))
print (np.shape(data_y))
#data_x, data_y = getseqs.balance_data(data_x, data_y, out_class)

# The model
input_layer = Input(shape=(max_seq,feat_size), dtype='float32', name="inputs")
#masked_input = Masking(mask_value=0)(input_layer)
dropout = Dropout(0.3,name="droput")(input_layer)
gru = Bidirectional(GRU(100, name="rnn",return_sequences=True), merge_mode='concat')(dropout)
adjacency = Lambda(Adjancency, output_shape=(max_seq,max_seq,200))(gru)
f = Flatten()(adjacency)
#hidden1 = TimeDistributed(Dense(200, activation='relu', name="hidden1"))(lstm)
#hidden2 = Dense(200, activation='relu', name="hidden2")(lstm)
#top = Dense(len(linkTypes), activation='softmax', name="top")(hidden2)
model = Model(input=input_layer, output=f)
model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])

print (model.predict(data_x))
sys.exit()
model.fit(data_x, data_y, batch_size=batch_size, nb_epoch=epochs, validation_split=val_split)


# Testing
entities, sequences,  _ = getseqs.get_testdata(test_path)
links = dict()
data_x, _, _ = getseqs.data_to_seq2lab_vector(links, entities, sequences, max_seq,
                                              types2idx, len(types),
                                              parentsTypes2dx, len(parentsTypes),
                                              linkTypes2idx, len(linkTypes))


print (np.shape(data_x))

predictions = model.predict(data_x,batch_size=batch_size,verbose=1)

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
