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

#train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_out = 'out/train'
test_out = 'out/test'

path = train_path
out_path = train_out

tnschema = anafora.get_schema()

# props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
#          "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
#          "Start-Interval", "Sub-Interval", "Time-Zone"]
#
# cond = " or ".join(["self::%s" % p for p in props])

for doc in os.listdir(path):
    for xmlfile in os.listdir(path + '/' + doc):
        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
        links = dict()
        stack = list()
        entity_list = dict()
        lend = -1
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = entity.find('./span').text.split(',')
            estart, eend = int(estart), int(eend)

            if estart - lend > 10 and lend > -1:
                stack = list()
                entity_list = dict()
            lend = eend

            etype = entity.find('./type').text
            eparentsType = entity.find('./parentsType').text
            entity_list[eid] = (estart, eend, etype, eparentsType)
            eproperties = entity.find('./properties')
            # Empty all links
            for prop in eproperties.findall('./*'):
                eproperties.remove(prop)

            ltype = ""
            stack_pointer = list()
            stack_pointer.extend(stack)
            while len(stack_pointer) > 0 and ltype == "":
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
