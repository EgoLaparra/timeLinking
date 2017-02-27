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
from keras.layers import Input, LSTM, TimeDistributed, Dense, Masking, Dropout
from keras.models import Model
import numpy as np

import getseqs


# LSTM Parameters
epochs = 10 # Number of epochs to cycle through data
batch_size = 100 # Train on this many examples at once
learning_rate = 0.001 # Learning rate
val_split = 0.25

# Get and process data
train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'

links, entities, sequences,  max_seq = getseqs.getdata(train_path)
(types, types2idx,
 parentsTypes, parentsTypes2dx,
 linkTypes, linkTypes2idx) = getseqs.build_vocabs(links, entities, sequences)
data_x, data_y, out_class = getseqs.data_to_seq2seq_vector(links, entities, sequences, max_seq,
                                        types2idx, len(types),
                                        parentsTypes2dx, len(parentsTypes),
                                        linkTypes2idx, len(linkTypes))
feat_size = 2 * len(types) + 2 * len(parentsTypes)

#data_x, data_y = getseqs.balance_data(data_x, data_y, out_class)

# The model

input_layer = Input(shape=(max_seq,feat_size), dtype='float32', name="inputs")
masked_input = Masking(mask_value=-1)(input_layer)
lstm = LSTM(70,return_sequences=True, name="lstm")(masked_input)
dropout = Dropout(0.3,name="droput")(lstm)
top = TimeDistributed(Dense(len(linkTypes), activation='softmax'), name="top")(lstm)
model = Model(input=input_layer, output=top)
model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(data_x, data_y, batch_size=batch_size, nb_epoch=epochs, validation_split=val_split)

# Testing
links, entities, sequences,  _ = getseqs.getdata(train_path)
data_x, _, _ = getseqs.data_to_seq2seq_vector(links, entities, sequences, max_seq,
                                        types2idx, len(types),
                                        parentsTypes2dx, len(parentsTypes),
                                        linkTypes2idx, len(linkTypes))

predictions = model.predict(data_x,batch_size=batch_size,verbose=1)
    
labels = list()
for s in predictions:
    for e in s:
        i = np.argmax(e)
        labels.append(linkTypes[i])

l = 0
for key in sequences.keys():
    for target in sorted(sequences[key]):
        for entity in sorted(sequences[key]):
            print (entities[entity][0],entities[entity][1],entities[entity][2],
                   entities[target][0],entities[target][1],entities[target][2],
                   labels[l])
            l += 1
        print ("")
    