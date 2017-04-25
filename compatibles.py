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

path = test_path
out_path = test_out

tnschema = anafora.get_schema()

# props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
#          "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
#          "Start-Interval", "Sub-Interval", "Time-Zone"]
#
# cond = " or ".join(["self::%s" % p for p in props])

for doc in os.listdir(path):
    for xmlfile in os.listdir(path + '/' + doc):
        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
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
            link = ''
            ltype = ''
            enext = entity.getnext()
            if enext is not None:
                nstart, nend = enext.find('./span').text.split(',')
                nstart, nend = int(nstart), int(nend)
                ntype = enext.find('./type').text
                nparentsType = enext.find('./parentsType').text
                if nstart - eend < 10:
                    ltype = get_relation(tnschema, etype, ntype)
                    if ltype != "":
                        link = enext.find('./id').text

            if link == '':
                eprev = entity.getprevious()
                if eprev is not None and eprev.find('./properties/*') is None:
                    ptstart, pend = eprev.find('./span').text.split(',')
                    ptstart, pend = int(ptstart), int(pend)
                    ptype = eprev.find('./type').text
                    pparentsType = eprev.find('./parentsType').text
                    if estart - pend < 10:
                        ltype = get_relation(tnschema, etype, ptype)
                        if ltype != "":
                            link = eprev.find('./id').text

            if ltype != '':
                si = etree.Element(ltype)
                if link != '':
                    si.text = link
                eproperties.append(si)

        for entity in axml.findall('.//entity'):
            etype = entity.find('./type').text
            eproperties = entity.find('./properties')
            if etype in tnschema:
                for relation in tnschema[etype]:
                    if tnschema[etype][relation][0]:
                        if eproperties.find('./' + relation) is None:
                            si = etree.Element(relation)
                            eproperties.append(si)

        if not os.path.exists(out_path + '/' + doc):
            os.makedirs(out_path + '/' + doc)
        axml.write(out_path + '/' + doc + '/' + xmlfile, pretty_print=True)
