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
            position = 0
            for entity in axml.findall('.//entity'):
                eid = entity.find('./id').text
                espan = entity.find('./span').text
                etype = entity.find('./type').text
                eparentsType = entity.find('./parentsType').text
                entities[eid] = (espan,etype,eparentsType,xmlfile,eid,position)
                position += 1
                for prop in entity.findall('./properties/*'):
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

        
def build_vocabs(links, entities, seqs):
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
                padd_x.fill(-1)
                seq_x.append(padd_x)
                padd_y = np.zeros(size_linkTypes)
                seq_y.append(padd_y)
                      
            x.append(np.array(seq_x))
            y.append(np.array(seq_y))
        
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
                        padd_x.fill(-1)
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
                    distance = entities[entity][5] - entities[target][5]
                    seq_x = np.append(
                                   np.concatenate(
                                      (etype_vector, eparentsType_vector,
                                       ttype_vector, tparentsType_vector),
                                      axis=0),
                                   distance)
 
                    if one_hot_labels:
                        linkType_vector = np.zeros(size_linkTypes)
                        linkType_vector[lType] = 1
                        seq_y = linkType_vector
                    else:
                        seq_y = lType
                
                      
                    x.append(np.array(seq_x))
                    y.append(np.array(seq_y))
        
    return np.array(x), np.array(y), out_class

    
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

        