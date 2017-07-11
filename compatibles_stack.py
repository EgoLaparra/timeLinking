#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:50:43 2017

@author: egoitz
"""

import getseqs, anafora, text2num
import dateutil.parser as dprs

from lxml import etree
import os



def get_relation(tnschema, parent, child):
    if parent in tnschema:
        for relation in tnschema[parent]:
            if relation != "parentsType":
                for validChild in tnschema[parent][relation][1]:
                    if validChild == child:
                        return relation
    return ""

rawpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'
dctpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/dct/'
train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
#rawpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'
#dctpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/dct/'
#train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
#te_path = 'in/715/'
te_path = 'in/794/'
train_out = 'out/train'
test_out = 'out/test'
#te_out = 'out/te'

path = test_path
out_path = test_out

tnschema = anafora.get_schema()
types = anafora.get_types()

import re
import sys
for doc in os.listdir(path):
    for xmlfile in os.listdir(path + '/' + doc):
        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
        rawfile = open(rawpath + '/' + doc, 'r')
        text = rawfile.read()
        rawfile.close()

        dctfile = open(dctpath + '/' + doc, 'r')
        dct = dctfile.read().rstrip()
        dctfile.close()
        try:
            dct = dprs.parse(dct)
            dctDayofWeek = dct.strftime('%A')
        except ValueError:
            dctDayofWeek = ""

        entities = dict()
        starts = dict()
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = map(int, entity.find('./span').text.split(','))
            etype = entity.find('./type').text
            eparentsType = entity.find('./parentsType')
            if eparentsType is not None:
                eparentsType = eparentsType.text
            else:
                eparentsType = tnschema[etype]["parentsType"]
                parentsType = etree.Element("parentsType")
                parentsType.text = eparentsType
                entity.append(parentsType)
            eproperties = entity.find('./properties')
            # Empty all links
            if eproperties is not None:
                for prop in eproperties.findall('./*'):
                    eproperties.remove(prop)
            else:
                prop = etree.Element("properties")
                entity.append(prop)
            if estart not in starts:
                starts[estart] = list()
            ent_values = (eid, estart, eend, etype, eparentsType)
            starts[estart].append(eid)
            entities[eid] = ent_values

        links = dict()
        stack = list()
        entity_list = dict()
        lend = -1
        for start in sorted(starts):
            for entity in starts[start]:
                (eid, estart, eend, etype, eparentsType) = entities[entity]
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
            estart, eend = map(int, entity.find('./span').text.split(','))
            eproperties = entity.find('./properties')
            if etype in tnschema:
                for relation in tnschema[etype]:
                    if relation != "parentsType":
                        span = "".join(text[estart:eend])
                        if relation == "Type":
                            ptype = span.title()
                            if ptype == "About":
                                ptype = "Approx"
                            if etype in types:
                                if span in types[etype]:
                                    ptype = types[etype][span]
                            ty = etree.Element(relation)
                            ty.text = ptype
                            eproperties.append(ty)
                        elif relation == "Value":
                            val = etree.Element(relation)
                            span = re.sub(r'^0(\d)', r'\1', re.sub(r'^0+', '0', span))
                            span = str(text2num.text2num(span))
                            val.text = span
                            eproperties.append(val)
                        elif re.search('Interval-Type',relation):
                            intervalemtpy = True
                            if eid in links:
                                if "Interval" in links[eid]:
                                    if links[eid]["Interval"] != "":
                                        intervalemtpy = False
                            if not intervalemtpy:
                                itype = etree.Element(relation)
                                itype.text = "Link"
                                eproperties.append(itype)
                            else:
                                itype = etree.Element(relation)
                                itype.text = "DocTime"
                                eproperties.append(itype)
                        elif relation == "Semantics":
                            sem = etree.Element(relation)
                            sem.text = "Standard"
                            eproperties.append(sem)
                        else:
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
                if etype == "Last":
                    semantics = eproperties.findall('./Semantics')[0]
                    for repint in eproperties.findall('./Repeating-Interval'):
                        if repint.text is not None:
                            (rid, rstart, rend, rtype, rparentsType) = entities[repint.text]
                            rspan = "".join(text[int(rstart):int(rend)])
                            if rspan.title() == dctDayofWeek:
                                semantics.text = "Newswire"

        if not os.path.exists(out_path + '/' + doc):
            os.makedirs(out_path + '/' + doc)
        axml.write(out_path + '/' + doc + '/' + xmlfile, pretty_print=True)
