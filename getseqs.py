#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:50:21 2017

@author: egoitz
"""
import sys
from lxml import etree
import os
import numpy as np

import anafora

props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
         "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
         "Start-Interval", "Sub-Interval", "Time-Zone"]

cond = "[" + " or ".join(["self::%s" % p for p in props]) + "]"

def getseq(seqs, node):
    seqkey = None
    for key in seqs.keys():
        if node in seqs[key]:
            seqkey = key
    return seqkey

    
def getdata(path):
    links = dict()
    entities = dict() 
    seqs = dict()
    max_seq = 0
    for doc in os.listdir(path):
        for xmlfile in os.listdir(path + '/' + doc):
            axml = etree.parse(path + '/' + doc + '/' + xmlfile)
            #entities["END"] = ("0","0","END","END","END",-1)
            position = 0
            for entity in axml.findall('.//entity'):
                eid = entity.find('./id').text
                espan = entity.find('./span').text
                etype = entity.find('./type').text
                eparentsType = entity.find('./parentsType').text
                entities[eid] = (espan,etype,eparentsType,xmlfile,eid,position)
                position += 1
                for prop in entity.xpath('./properties/*' + cond):
                    ptag = prop.tag
                    ptext = prop.text
                    if ptext is not None:
                        lids = axml.xpath('.//entity/id[text()="'+ ptext +'"]')
                        if len(lids) == 1:
                            lid = lids[0].text
                            if eid not in links:
                                links[eid] = dict()
                            links[eid][lid] = ptag
                            eseqkey = getseq(seqs, eid)
                            lseqkey = getseq(seqs, lid)
                            if eseqkey is None:
                                if lseqkey is None:
                                    newseq = set()
                                    newseq.add(eid)
                                    newseq.add(lid)
                                    seqs[eid] = newseq
                                    if len(seqs[eid]) > max_seq:
                                        max_seq = len(seqs[eid])
                                else:
                                    seqs[lseqkey].add(eid)
                                    if len(seqs[lseqkey]) > max_seq:
                                        max_seq = len(seqs[lseqkey])
                            else:
                                if lseqkey is None:
                                    seqs[eseqkey].add(lid)
                                    if len(seqs[eseqkey]) > max_seq:
                                        max_seq = len(seqs[eseqkey])
                                elif eseqkey != lseqkey:
                                    for node in seqs[lseqkey]:
                                        seqs[eseqkey].add(node)
                                    del seqs[lseqkey]
                                    if len(seqs[eseqkey]) > max_seq:
                                        max_seq = len(seqs[eseqkey])

    return links, entities, seqs, max_seq

    
def get_testdata(path):
    links = dict()
    entities = dict() 
    seqs = dict()
    max_seq = 0
                
    for doc in os.listdir(path):
        for xmlfile in os.listdir(path + '/' + doc):
            axml = etree.parse(path + '/' + doc + '/' + xmlfile)
            #entities["END"] = ("0","0","END","END","END",-1)
            position = 0
            for entity in axml.findall('.//entity'):
                eid = entity.find('./id').text
                espan = entity.find('./span').text
                ebegin, eend = espan.split(',')
                etype = entity.find('./type').text
                eparentsType = entity.find('./parentsType').text
                entities[eid] = (espan,etype,eparentsType,xmlfile,eid,position)
                position += 1
                enext = entity.getnext()
                if enext is not None:
                    lid = enext.find('./id').text
                    lspan = enext.find('./span').text
                    lbegin, lend = lspan.split(',')
                    if int(lbegin) - int(eend) < 10:
                    #if True:
                        eseqkey = getseq(seqs, eid)
                        lseqkey = getseq(seqs, lid)
                        if eseqkey is None:
                            if lseqkey is None:
                                newseq = set()
                                newseq.add(eid)
                                newseq.add(lid)
                                seqs[eid] = newseq
                                if len(seqs[eid]) > max_seq:
                                    max_seq = len(seqs[eid])
                            else:
                                seqs[lseqkey].add(eid)
                                if len(seqs[lseqkey]) > max_seq:
                                    max_seq = len(seqs[lseqkey])
                        else:
                            if lseqkey is None:
                                seqs[eseqkey].add(lid)
                                if len(seqs[eseqkey]) > max_seq:
                                    max_seq = len(seqs[eseqkey])
                            elif eseqkey != lseqkey:
                                for node in seqs[lseqkey]:
                                    seqs[eseqkey].add(node)
                                del seqs[lseqkey]
                                if len(seqs[eseqkey]) > max_seq:
                                    max_seq = len(seqs[eseqkey])

    return entities, seqs, max_seq
        
def build_vocabs(links, entities, seqs):
    types = list()
    types.append('unk')
    types_to_idx = dict()
    types_to_idx['unk'] = 0
    parentsTypes = list()
    parentsTypes.append('unk')
    parentsTypes_to_idx = dict()
    linkTypes = list()
    linkTypes.append('O')
    linkTypes_to_idx = dict()
    linkTypes_to_idx['O'] = 0
    
    for key in seqs.keys():
        for target in sorted(seqs[key]):
            for entity in sorted(seqs[key]):
                if entities[entity][1] not in types:
                    types_to_idx[entities[entity][1]] = len(types)
                    types.append(entities[entity][1])
                if entities[target][1] not in types:
                    types_to_idx[entities[target][1]] = len(types)
                    types.append(entities[target][1])
                if entities[entity][2] not in parentsTypes:
                    parentsTypes_to_idx[entities[entity][2]] = len(parentsTypes)
                    parentsTypes.append(entities[entity][2])
                if entities[target][2] not in parentsTypes:
                    parentsTypes_to_idx[entities[target][2]] = len(parentsTypes)
                    parentsTypes.append(entities[target][2])
                if target in links.keys() and entity in links[target].keys():
                    if links[target][entity] not in linkTypes:
                        linkTypes_to_idx[links[target][entity]] = len(linkTypes)
                        linkTypes.append(links[target][entity])
    
    return types, types_to_idx, parentsTypes, parentsTypes_to_idx, linkTypes, linkTypes_to_idx


def build_vocabs_s2s(links, entities, seqs,transitions):
    types = list()
    types.append('unk')
    types_to_idx = dict()
    types_to_idx['unk'] = 0
    parentsTypes = list()
    parentsTypes.append('unk')
    parentsTypes_to_idx = dict()
    parentsTypes_to_idx['unk'] = 0
    linkTypes = list()
    linkTypes.append('O')
    linkTypes_to_idx = dict()
    linkTypes_to_idx['O'] = 0
    
    for key in seqs.keys():
        for target in sorted(seqs[key]):
            for entity in sorted(seqs[key]):
                if entities[entity][1] not in types:
                    types_to_idx[entities[entity][1]] = len(types)
                    types.append(entities[entity][1])
                if entities[target][1] not in types:
                    types_to_idx[entities[target][1]] = len(types)
                    types.append(entities[target][1])
                if entities[entity][2] not in parentsTypes:
                    parentsTypes_to_idx[entities[entity][2]] = len(parentsTypes)
                    parentsTypes.append(entities[entity][2])
                if entities[target][2] not in parentsTypes:
                    parentsTypes_to_idx[entities[target][2]] = len(parentsTypes)
                    parentsTypes.append(entities[target][2])
                if target in links.keys() and entity in links[target].keys():
                    if links[target][entity] not in linkTypes:
                        linkTypes_to_idx[links[target][entity]] = len(linkTypes)
                        linkTypes.append(links[target][entity])
    
    return types, types_to_idx, parentsTypes, parentsTypes_to_idx, linkTypes, linkTypes_to_idx

    
def data_to_seq2seq_vector(links, entities, seqs, max_seq,
                   types_to_idx, size_types, 
                   parentsTypes_to_idx, size_parentTypes,
                   linkTypes_to_idx, size_linkTypes):
    
    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        for target in sorted(seqs[key]):
            seq_x = list()
            seq_y = list()
            for entity in sorted(seqs[key]):
                etype = 0
                if entities[entity][1] in types_to_idx.keys():
                    etype = types_to_idx[entities[entity][1]]
                eparentsType = 0
                if entities[entity][2] in parentsTypes_to_idx.keys():
                    eparentsType = parentsTypes_to_idx[entities[entity][2]]
                ttype = 0
                if entities[target][1] in types_to_idx.keys():
                    ttype = types_to_idx[entities[target][1]]
                tparentsType = 0
                if entities[target][2] in parentsTypes_to_idx:
                    tparentsType = parentsTypes_to_idx[entities[target][2]]
                lType = -1
                if target in links.keys() and entity in links[target].keys():
                    lType = linkTypes_to_idx[links[target][entity]]
                else:
                    lType = linkTypes_to_idx['O']
                
                etype_vector = np.zeros(size_types)
                etype_vector[etype] = 1
                eparentsType_vector = np.zeros(size_parentTypes)
                eparentsType_vector[eparentsType] = 1
                ttype_vector = np.zeros(size_types)
                ttype_vector[ttype] = 1
                tparentsType_vector = np.zeros(size_parentTypes)
                tparentsType_vector[tparentsType] = 1
                seq_x.append(np.concatenate(
                            (etype_vector, eparentsType_vector,
                             ttype_vector, tparentsType_vector),
                             axis=0))
                    
                linkType_vector = np.zeros(size_linkTypes)
                linkType_vector[lType] = 1
                seq_y.append(linkType_vector)

            out_idx = linkTypes_to_idx['O']
            out_vector = np.zeros(size_linkTypes)
            out_vector[out_idx] = 1
            if np.all(seq_y==out_vector):
                out_class.append(len(y))
            feat_size = 2 * size_types + 2 * size_parentTypes
            while len(seq_x) < max_seq:
                padd_x = np.empty(feat_size)
                padd_x.fill(0)
                seq_x.append(padd_x)
                #padd_y = np.zeros(size_linkTypes)
                #seq_y.append(padd_y)
                      
            x.append(np.array(seq_x))
            y.append(np.array(seq_y))
        
    return np.array(x), np.array(y), out_class



def data_to_seq2seq(links, entities, seqs, max_seq,
                                  transitions, max_tran_seq,
                                  types_to_idx, size_types,
                                  parentsTypes_to_idx, size_parentTypes,
                                  tran_to_idx, size_transitions):
    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        seq_x = list()
        for entity in sorted(seqs[key]):

            etype = 0
            if entities[entity][1] in types_to_idx.keys():
                etype = types_to_idx[entities[entity][1]]
            eparentsType = 0
            if entities[entity][2] in parentsTypes_to_idx.keys():
                eparentsType = parentsTypes_to_idx[entities[entity][2]]

            seq_x.append(etype)

        feat_size = size_types
        while len(seq_x) < max_seq:
            seq_x.append(-1)

        x.append(np.array(seq_x))

    for transition_seq in transitions:
        seq_y = list()
        for transition in transition_seq:
            tranOp = 0
            if transition in tran_to_idx.keys():
                tranOp = tran_to_idx[transition]

            tran_vector = np.zeros(size_transitions)
            tran_vector[tranOp] = 1
            seq_y.append(tran_vector)

        while len(seq_y) < max_tran_seq:
            padd_y = np.zeros(size_transitions)
            padd_y[0] = 1
            seq_y.append(padd_y)

        y.append(np.array(seq_y))

    return np.array(x), np.array(y), out_class


def data_to_seq2seq_single_vector(links, entities, seqs, max_seq,
                                  transitions, max_tran_seq,
                                   types_to_idx, size_types, 
                                   parentsTypes_to_idx, size_parentTypes,
                                   tran_to_idx, size_transitions):
        
    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        seq_x = list()
        for entity in sorted(seqs[key]):

            etype = 0
            if entities[entity][1] in types_to_idx.keys():
                etype = types_to_idx[entities[entity][1]]
            eparentsType = 0
            if entities[entity][2] in parentsTypes_to_idx.keys():
                eparentsType = parentsTypes_to_idx[entities[entity][2]]
                    
            etype_vector = np.zeros(size_types)
            etype_vector[etype] = 1
            eparentsType_vector = np.zeros(size_parentTypes)
            eparentsType_vector[eparentsType] = 1
            seq_x.append(np.concatenate(
                        (etype_vector, eparentsType_vector),
                         axis=0))
                  
        feat_size = size_types + size_parentTypes
        while len(seq_x) < max_seq:
            padd_x = np.empty(feat_size)
            padd_x.fill(0)
            seq_x.append(padd_x)
            
        x.append(np.array(seq_x))
            
    for transition_seq in transitions:
        seq_y = list()
        for transition in transition_seq:
            tranOp = 0
            if transition in tran_to_idx.keys():
                tranOp = tran_to_idx[transition]
 
            tran_vector = np.zeros(size_transitions)
            tran_vector[tranOp] = 1
            seq_y.append(tran_vector)
                  
        while len(seq_y) < max_tran_seq:
            padd_y = np.zeros(size_transitions)
            padd_y[0] = 1
            seq_y.append(padd_y)
         
        y.append(np.array(seq_y))
        
    return np.array(x), np.array(y), out_class

    
def data_to_seq2graph(links, entities, seqs, max_seq,
                   types_to_idx, size_types, 
                   parentsTypes_to_idx, size_parentTypes,
                   linkTypes_to_idx, size_linkTypes):

    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        seq_x = list()
        seq_y = list()
        for target in sorted(seqs[key]):
            ttype = 0
            if entities[target][1] in types_to_idx.keys():
                ttype = types_to_idx[entities[target][1]]
            tparentsType = 0
            if entities[target][2] in parentsTypes_to_idx:
                tparentsType = parentsTypes_to_idx[entities[target][2]]
            ttype_vector = np.zeros(size_types)
            ttype_vector[ttype] = 1
            tparentsType_vector = np.zeros(size_parentTypes)
            tparentsType_vector[tparentsType] = 1
            seq_x.append(np.concatenate(
                        (ttype_vector, tparentsType_vector),
                        axis=0))
                        
            target_y = list()
            for entity in sorted(seqs[key]):
                lType = -1
                if target in links.keys() and entity in links[target].keys():
                    lType = linkTypes_to_idx[links[target][entity]]
                else:
                    lType = linkTypes_to_idx['O']
                    out_class.append(len(y))
                linkType_vector = np.zeros(size_linkTypes)
                linkType_vector[lType] = 1
                target_y.append(linkType_vector)
            while len(target_y) < max_seq:
                padd_y = np.empty(size_linkTypes)
                padd_y.fill(0)
                target_y.append(padd_y)
            seq_y.append(target_y)

            
        feat_size = size_types + size_parentTypes
        while len(seq_x) < max_seq:
            padd_x = np.empty(feat_size)
            padd_x.fill(0)
            seq_x.append(padd_x)
    
        while len(seq_y) < max_seq:
            target_y = list()
            while len(target_y) < max_seq:
                padd_y = np.empty(size_linkTypes)
                padd_y.fill(0)
                target_y.append(padd_y)
            seq_y.append(target_y)                
        
        x.append(seq_x)
        y.append(seq_y)
            
    return np.array(x), np.array(y), out_class
               
    
def data_to_seq2lab_vector(links, entities, seqs, max_seq,
                   types_to_idx, size_types, 
                   parentsTypes_to_idx, size_parentTypes,
                   linkTypes_to_idx, size_linkTypes):
    
    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        for target in sorted(seqs[key]):
            for entity in sorted(seqs[key]):
                if target != entity:
                    seq_y = list()
                    etype = 0
                    if entities[entity][1] in types_to_idx.keys():
                        etype = types_to_idx[entities[entity][1]]
                    eparentsType = 0
                    if entities[entity][2] in parentsTypes_to_idx.keys():
                        eparentsType = parentsTypes_to_idx[entities[entity][2]]
                    ttype = 0
                    if entities[target][1] in types_to_idx.keys():
                        ttype = types_to_idx[entities[target][1]]
                    tparentsType = 0
                    if entities[target][2] in parentsTypes_to_idx:
                        tparentsType = parentsTypes_to_idx[entities[target][2]]
                    lType = -1
                    if target in links.keys() and entity in links[target].keys():
                        lType = linkTypes_to_idx[links[target][entity]]
                    else:
                        lType = linkTypes_to_idx['O']
                        out_class.append(len(y))
    
                    seq_x = list()
                    for context in sorted(seqs[key]):
                        ctype = 0
                        if entities[context][1] in types_to_idx.keys():
                            ctype = types_to_idx[entities[context][1]]
                        cparentsType = 0
                        if entities[entity][2] in parentsTypes_to_idx.keys():
                            cparentsType = parentsTypes_to_idx[entities[context][2]]
                
                        ctype_vector = np.zeros(size_types)
                        ctype_vector[ctype] = 1
                        cparentsType_vector = np.zeros(size_parentTypes)
                        cparentsType_vector[cparentsType] = 1
                        etype_vector = np.zeros(size_types)
                        etype_vector[etype] = 1
                        eparentsType_vector = np.zeros(size_parentTypes)
                        eparentsType_vector[eparentsType] = 1
                        ttype_vector = np.zeros(size_types)
                        ttype_vector[ttype] = 1
                        tparentsType_vector = np.zeros(size_parentTypes)
                        tparentsType_vector[tparentsType] = 1
                        seq_x.append(np.concatenate(
                                    (ctype_vector, cparentsType_vector, 
                                     etype_vector, eparentsType_vector, 
                                     ttype_vector, tparentsType_vector),
                                    axis=0))
                        
                    linkType_vector = np.zeros(size_linkTypes)
                    linkType_vector[lType] = 1
                    seq_y = linkType_vector
                
                    feat_size = 3 * size_types + 3 * size_parentTypes
                    while len(seq_x) < max_seq:
                        padd_x = np.empty(feat_size)
                        padd_x.fill(0)
                        seq_x.append(padd_x)

                    x.append(np.array(seq_x))
                    y.append(np.array(seq_y))

    return np.array(x), np.array(y), out_class
    

def data_to_pair2lab_vector(links, entities, seqs, max_seq,
                            types_to_idx, size_types, 
                            parentsTypes_to_idx, size_parentTypes,
                            linkTypes_to_idx, size_linkTypes,
                            one_hot_labels=True):
    
    x = list()
    y = list()
    out_class = list()
    for key in seqs.keys():
        for target in sorted(seqs[key]):
            for entity in sorted(seqs[key]):
                if target != entity:
                    seq_x = list()
                    seq_y = list()
                    
                    etype = 0
                    if entities[entity][1] in types_to_idx.keys():
                        etype = types_to_idx[entities[entity][1]]
                    eparentsType = 0
                    if entities[entity][2] in parentsTypes_to_idx.keys():
                        eparentsType = parentsTypes_to_idx[entities[entity][2]]
                    ttype = 0
                    if entities[target][1] in types_to_idx.keys():
                        ttype = types_to_idx[entities[target][1]]
                    tparentsType = 0
                    if entities[target][2] in parentsTypes_to_idx:
                        tparentsType = parentsTypes_to_idx[entities[target][2]]
                    lType = -1
                    if target in links.keys() and entity in links[target].keys():
                        lType = linkTypes_to_idx[links[target][entity]]
                    else:
                        lType = linkTypes_to_idx['O']
                        out_class.append(len(y))
    
                                           
                    etype_vector = np.zeros(size_types)
                    etype_vector[etype] = 1
                    eparentsType_vector = np.zeros(size_parentTypes)
                    eparentsType_vector[eparentsType] = 1
                    ttype_vector = np.zeros(size_types)
                    ttype_vector[ttype] = 1
                    tparentsType_vector = np.zeros(size_parentTypes)
                    tparentsType_vector[tparentsType] = 1
                    distanceTokens = entities[entity][5] - entities[target][5]
                    targetbegin = int(entities[target][0].split(',')[0])
                    entitybegin = int(entities[entity][0].split(',')[0])
                    distanceChars = entitybegin - targetbegin
                    seq_x = np.append(
                                   np.concatenate(
                                      (etype_vector, eparentsType_vector,
                                       ttype_vector, tparentsType_vector),
                                      axis=0),
                                   #distanceTokens)
                                   distanceChars)
 
                    if one_hot_labels:
                        linkType_vector = np.zeros(size_linkTypes)
                        linkType_vector[lType] = 1
                        seq_y = linkType_vector
                    else:
                        seq_y = lType
                
                      
                    x.append(np.array(seq_x))
                    y.append(np.array(seq_y))
        
    return np.array(x), np.array(y), out_class

 
def split_x(data_x):
    data_x_context, data_x_entity, data_x_target = list(),list(),list()
    for seq in data_x:
        context = list()
        for vect  in seq:
            ctx, enty, tgt = np.split(vect, 3)
            context.append(ctx)
        data_x_context.append(np.array(context))
        data_x_entity.append(enty)
        data_x_target.append(tgt)
        
    return np.array(data_x_context), np.array(data_x_entity), np.array(data_x_target)
    
def balance_data(data_x, data_y, out_class):
    sample_outs = np.random.choice(out_class, 1000)
    balanced_data_x = list()
    balanced_data_y = list()
    for i in range(0, len(data_y)):
        if i not in out_class or i in sample_outs:
            balanced_data_x.append(data_x[i])
            balanced_data_y.append(data_y[i])
    return np.array(balanced_data_x), np.array(balanced_data_y)
    
def weight_classes(classes):
    weights = dict()
    weights[0] = 1 / 200
    for i in range(1,len(classes)):
        weights[i] = 1 - 1/200
    return weights
    
def print_outputs(test_path, out_path, outputs):
    for doc in os.listdir(test_path):
        for xmlfile in os.listdir(test_path + '/' + doc):
            axml = etree.parse(test_path + '/' + doc + '/' + xmlfile)
            for entity in axml.findall('.//entity'):
                eproperties = entity.find('./properties')
                # Empty all links
                for prop in eproperties.findall('./*'):
                    eproperties.remove(prop)
        
            for target in outputs[xmlfile]:
                targetProps = axml.xpath('.//entity[id/text()="' + target + '"]/properties')[0]
                for entity in outputs[xmlfile][target]:
                    link = outputs[xmlfile][target][entity][0]
                    prob = outputs[xmlfile][target][entity][1]
                    l = etree.Element(link)
                    l.text = entity
                    targetProps.append(l)
        
        if not os.path.exists(out_path + '/' + doc):
            os.makedirs(out_path + '/' + doc)
        axml.write(out_path + '/' + doc + '/' + xmlfile, pretty_print=True)

        
def get_transitions(links, entities, seqs):
    transitions = []
    entitylists = []
    newlinks = []
    transOp = ["Stop","LStep", "RStep", "Pass"]
    trans2idx = dict()
    trans2idx["Stop"] = 0
    trans2idx["LStep"] = 1
    trans2idx["RStep"] = 2
    trans2idx["Pass"] = 3
    max_trans = 0
    for key in seqs.keys():
        arcs = list()
        transition = list()
        entitylist = list()
        for entity in seqs[key]:
            begin = entities[entity][0].split(',')[0]
            entitylist.append((entity, int(begin)))
        entitylist = sorted(entitylist, key=lambda x: x[1])

                
        passes = []
        for e in range(0,len(entitylist)):
            isteps = []
            for l in range(e-1, -1, -1):
                if entitylist[e][0] in links and entitylist[l][0] in links[entitylist[e][0]]:
                    if len(passes) > 0:
                        transition.extend(passes)
                        passes = []
                    if len(isteps) > 0:
                        transition.extend(isteps)
                        isteps=[]
                    t = "L" + links[entitylist[e][0]][entitylist[l][0]]
                    if t not in transOp:
                        trans2idx[t] = len(transOp)
                        transOp.append(t)
                    arcs.append(entitylist[e][0] + " " + t + " " + entitylist[l][0])
                    transition.append(t)
                else:
                    isteps.append("LStep")
            rsteps = []
            for r in range(e+1, len(entitylist)):
                if entitylist[e][0] in links and entitylist[r][0] in links[entitylist[e][0]]:
                    if len(passes) > 0:
                        transition.extend(passes)
                        passes = []
                    if len(rsteps) > 0:
                        transition.extend(rsteps)
                        rsteps=[]
                    t = "R" + links[entitylist[e][0]][entitylist[r][0]]
                    if t not in transOp:
                        trans2idx[t] = len(transOp)
                        transOp.append(t)
                    arcs.append(entitylist[e][0] + " " + t + " " + entitylist[r][0])
                    transition.append(t)
                else:
                    rsteps.append("RStep") 
            passes.append("Pass")
        transition.append("Stop")

        if len(transition) > max_trans:
            max_trans = len(transition)
        transitions.append(transition)
        entitylists.append(entitylist)
        newlinks.append(links)

    return entitylists, transitions, newlinks, max_trans, transOp, trans2idx
        
import re   
def build_graph(entitylists, entities, transitions):
    outputs = dict()
    for i in range(0, len(transitions)):
        entitylist = entitylists[i]            
        e = 0
        l = e-1
        r = e+1   
        for t in transitions[i]:
            if e >= len(entitylist):
                continue
            if t == "Pass":
                e += 1
                l = e-1
                r = e+1
            elif t == "RStep":
                r += 1
            elif t == "LStep":
                l -= 1
            elif re.match(r'^L',t):
                if l < 0:
                    continue
                t = re.sub(r'^L','',t)
                target = entitylist[e][0]
                xmlfile = entities[target][3]
                if xmlfile not in outputs:
                    outputs[xmlfile] = dict()
                targetid = entities[target][4]
                if targetid not in outputs[xmlfile]:
                    outputs[xmlfile][targetid] = dict()
                outputs[xmlfile][targetid][entitylist[l][0]] = [t, 1.]
                l -= 1
            elif re.match(r'^R',t):
                if r >= len(entitylist):
                    continue
                t = re.sub(r'^R','',t)
                target = entitylist[e][0]
                xmlfile = entities[target][3]
                if xmlfile not in outputs:
                    outputs[xmlfile] = dict()
                targetid = entities[target][4]
                if targetid not in outputs[xmlfile]:
                    outputs[xmlfile][targetid] = dict()
                outputs[xmlfile][targetid][entitylist[r][0]] = [t, 1.]
                r += 1
    return outputs

    
def get_transitions_typereduce(links, entities, seqs):
    transitions = []
    entitylists = []
    newlinks = []
    transOp = ["Stop","Shift"]
    trans2idx = dict()
    trans2idx["Stop"] = 0
    trans2idx["Shift"] = 1
    max_trans = 0
    for key in seqs.keys():
        transition = list()
        stack = dict()
        stack_roots = list()
        entitylist = list()
        for entity in seqs[key]:
            begin = entities[entity][0].split(',')[0]
            entitylist.append((entity, int(begin)))
        entitylist = sorted(entitylist, key=lambda x: x[1])
        queue = list()
        queue.extend(entitylist)
        queue.reverse()
        
        while len(queue) > 0 or len(stack_roots) > 1:
            if len(stack_roots) <= 1:
                q = queue.pop()[0]
                transition.append('Shift')        
                stack[q] = list()
                stack_roots.append(q)
            else:
                remain_roots = list()
                while len(stack_roots) > 1:
                    r = stack_roots.pop()
                    t = ""
                    stack_pointer = list()
                    stack_pointer.extend(stack_roots)
                    stack_pointer.reverse()
                    while len(stack_pointer) > 0:
                        s = stack_pointer.pop()
                        if r in links and s in links[r]:
                            t = "TypeReduce2-" + links[r][s]
                            if s in stack_roots:
                                stack_roots.remove(s)
                            remain_roots.append(r)
                            stack[r].append(s)
                        elif s in links and r in links[s]:
                            t = "TypeReduce1-" + links[s][r]
                            stack[s].append(r)
                            remain_roots.append(s)
                        else:
                            stack_pointer.reverse()
                            stack_pointer.extend(stack[s])
                            stack_pointer.reverse()
                    if t == "":
                        remain_roots.append(r)
                        if len(queue) > 0:
                            q = queue.pop()[0]
                            transition.append('Shift')        
                            stack[q] = list()
                            remain_roots.append(q)
                    else:
                        if t not in transOp:
                                trans2idx[t] = len(transOp)
                                transOp.append(t)
                        transition.append(t)
                stack_roots = remain_roots

        transition.append("Stop")
        if len(transition) > max_trans:
            max_trans = len(transition)
        transitions.append(transition)
        entitylists.append(entitylist)
        newlinks.append(links)

    return entitylists, transitions, newlinks, max_trans, transOp, trans2idx

def compatible(tnschema, parent, child, relation):

    iscompatible = False
    validchilds = list()
    if parent[1] in tnschema:
        if relation in tnschema[parent[1]]:
            validchilds = tnschema[parent[1]][relation][1]
    if child[1] in validchilds:
        iscompatible = True
    return iscompatible


def build_graph_typereduce(entitylists, entities, transitions):

    tnschema = anafora.get_schema()
    outputs = dict()
    for i in range(0, len(transitions)):
        entitylist = entitylists[i]
        queue = entitylist
        queue.reverse()
        stack = dict()
        stack_roots = list()
        for t in transitions[i]:
            if t == "Shift" and len(queue) > 0:
                q = queue.pop()[0]
                stack[q] = list()
                stack_roots.append(q)
            elif re.match(r'^TypeReduce2',t):
                t = re.sub(r'^TypeReduce2-','',t)
                entity = stack_roots.pop()
                xmlfile = entities[entity][3]
                if xmlfile not in outputs:
                    outputs[xmlfile] = dict()
                stack_pointer = list()
                stack_pointer.append(entity)
                stack_pointer.reverse()
                found = False
                while len(stack_pointer) > 0:
                    s = stack_pointer.pop()
                    for r in stack_roots:
                        if compatible(tnschema, entities[s], entities[r], t):
                            if s not in outputs[xmlfile]:
                                outputs[xmlfile][s] = dict()
                            outputs[xmlfile][s][r] = [t, 1.]
                            stack[s].append(r)
                            stack_pointer = []
                            found = True
                            continue
                if not found:
                    stack_pointer.reverse()
                    stack_pointer.extend(stack[s])
                    stack_pointer.reverse()

            elif re.match(r'^TypeReduce1',t):
                t = re.sub(r'^TypeReduce1-','',t)
                entity = stack_roots.pop()
                xmlfile = entities[entity][3]
                if xmlfile not in outputs:
                    outputs[xmlfile] = dict()
                stack_pointer = list()
                stack_pointer.extend(stack_roots)
                stack_pointer.reverse()
                while len(stack_pointer) > 0:
                    s = stack_pointer.pop()
                    if compatible(tnschema, entities[s],entities[entity],t):
                        if s not in outputs[xmlfile]:
                            outputs[xmlfile][s] = dict()
                        outputs[xmlfile][s][entity] = [t, 1.]
                        stack[s].append(entity)
                        stack_pointer = []
                    else:
                        stack_pointer.reverse()
                        stack_pointer.extend(stack[s])
                        stack_pointer.reverse()

    return outputs

# train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'

#links, entities, sequences,  max_seq = getdata(train_path)
#ent,tran,nl,mt,to,ti = get_transitions_typereduce(links,entities,sequences)
# for t in tran:
#     print (t)
#ent,tran,nl = get_transitions(links,entities,sequences)
#outputs = build_graph_typereduce(ent,entities,tran)

# for r in links:
#     for e in links[r]:
#         print ("D",r,e,links[r][e])
#
# for f in outputs:
#     for t in outputs[f]:
#         for e in outputs[f][t]:
#             print ("T",t,e,outputs[f][t][e][0])