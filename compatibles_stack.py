#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:50:43 2017

@author: egoitz
"""

import getseqs, anafora

from lxml import etree
import os

def get_relation(tnschema, parent, child):

    if parent in tnschema:
        for relation in tnschema[parent]:
           for validChild in tnschema[parent][relation][1]:
                if validChild == child:
                    return relation
    return ""

rawpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'
#train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_out = 'out/train'
test_out = 'out/test'

path = train_path
out_path = train_out

tnschema = anafora.get_schema()

import re
import sys
for doc in os.listdir(path):
    for xmlfile in os.listdir(path + '/' + doc):
        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
        rawfile = open(rawpath + '/' + doc, 'r')
        text = rawfile.read()
        rawfile.close()

        entities = dict()
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = entity.find('./span').text.split(',')
            estart, eend = int(estart), int(eend)
            etype = entity.find('./type').text
            eparentsType = entity.find('./parentsType').text
            eproperties = entity.find('./properties')
            # Empty all links
            for prop in eproperties.findall('./*'):
                eproperties.remove(prop)
            if estart not in entities:
                entities[estart] = list()
            ent_values = (eid, estart, eend, etype, eparentsType)
            entities[estart].append(ent_values)

        if re.search('PRI19980205.2000.1998',doc):
            for s in sorted(entities):
                for e in entities[s]:
                    end = e[2]
                    print (e, re.sub(' ', '_', "".join(text[s:end])))


        links = dict()
        stack = list()
        entity_list = dict()
        lend = -1
        for start in sorted(entities):
            for entity in entities[start]:
                (eid, estart, eend, etype, eparentsType) = entity
                if estart - lend > 10 and lend > -1:
                    stack = list()
                    entity_list = dict()
                lend = eend
                entity_list[eid] = (estart, eend, etype, eparentsType)
                ltype = ""
                stack_pointer = list()
                stack_pointer.extend(stack)
                while len(stack_pointer) > 0:
                    s = stack_pointer.pop()
                    stype = entity_list[s][2]
                    ltype = get_relation(tnschema, etype, stype)
                    if ltype != '':
                        if eid not in links:
                            links[eid] = dict()
                        if ltype not in links[eid]:
                            links[eid][ltype] = list()
                            links[eid][ltype].append(s)
                    else:
                        ltype = get_relation(tnschema, stype, etype)
                        if ltype != '':
                            if s not in links:
                                links[s] = dict()
                            if ltype not in links[s]:
                                links[s][ltype] = list()
                                links[s][ltype].append(eid)
                stack.append(eid)



        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            etype = entity.find('./type').text
            eproperties = entity.find('./properties')
            if etype in tnschema:
                for relation in tnschema[etype]:
                    notnull = False
                    if eid in links:
                        if relation in links[eid]:
                            for child in links[eid][relation]:
                                si = etree.Element(relation)
                                si.text = child
                                eproperties.append(si)
                                notnull = True
                    if tnschema[etype][relation][0] and not notnull:
                        if eproperties.find('./' + relation) is None:
                            si = etree.Element(relation)
                            eproperties.append(si)


        if not os.path.exists(out_path + '/' + doc):
            os.makedirs(out_path + '/' + doc)
        axml.write(out_path + '/' + doc + '/' + xmlfile, pretty_print=True)
