#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:50:43 2017

@author: egoitz
"""

from lxml import etree
import os
#import sys
#sys.path.append('/home/egoitz/Tools/time/anaforatools')
#import anafora

#adata = anafora.AnaforaData()
#afile = adata.from_file('/home/egoitz/Desktop/prueba/gold/APW19980807.0261.TimeNorm.gold.completed.xml')
#axml = afile.xml

#train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_out = 'out/train'
test_out = 'out/test'

path = test_path
out_path = test_out

props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
         "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
         "Start-Interval", "Sub-Interval", "Time-Zone"]

cond = " or ".join(["self::%s" % p for p in props])

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
            # Rules for Sub-Interval links
            link = ''
            ltype = ''
            if eparentsType == "Interval" or eparentsType == "Repeating-Interval":
                ltype = 'Sub-Interval'
                enext = entity.getnext()
                if enext is not None:
                    nstart, nend = enext.find('./span').text.split(',')
                    nstart, nend = int(nstart), int(nend)
                    ntype = enext.find('./type').text
                    nparentsType = enext.find('./parentsType').text
                    if nstart - eend == 1 and nparentsType == "Repeating-Interval":
                        link = enext.find('./id').text
                if link == '':
                    eprev = entity.getprevious()
                    if eprev is not None and eprev.find('./properties/*') is None:
                        ptstart, pend = eprev.find('./span').text.split(',')
                        ptstart, pend = int(ptstart), int(pend)
                        ptype = eprev.find('./type').text
                        pparentsType = eprev.find('./parentsType').text
                        if estart - pend == 1 and pparentsType == "Repeating-Interval":
                            link = eprev.find('./id').text
           # Rules for Repeating-Interval links
            elif eparentsType == "Operator":
                ltype = 'Repeating-Interval'
                enext = entity.getnext()
                if enext is not None:
                    nstart, nend = enext.find('./span').text.split(',')
                    nstart, nend = int(nstart), int(nend)
                    ntype = enext.find('./type').text
                    nparentsType = enext.find('./parentsType').text
                    if nstart - eend == 1 and nparentsType == "Repeating-Interval":
                        link = enext.find('./id').text
                if link == '':
                    eprev = entity.getprevious()
                    if eprev is not None:
                        ptstart, pend = eprev.find('./span').text.split(',')
                        ptstart, pend = int(ptstart), int(pend)
                        ptype = eprev.find('./type').text
                        pparentsType = eprev.find('./parentsType').text
                        if estart - pend == 1 and pparentsType == "Repeating-Interval":
                            link = eprev.find('./id').text
            # Rules for Duration parentTypes
            elif eparentsType == "Duration":
                eprev = entity.getprevious()
                if eprev is not None:
                    pstart, pend = eprev.find('./span').text.split(',')
                    pstart, pend = int(pstart), int(pend)
                    ptype = eprev.find('./type').text
                    pparentsType = eprev.find('./parentsType').text
                    if estart - pend == 1 and pparentsType == "Other":
                        ltype = ptype
                        link = eprev.find('./id').text            
            if ltype != '':
                si = etree.Element(ltype)
                if link != '':
                    si.text = link
                eproperties.append(si)

        if not os.path.exists(out_path + '/' + doc):
            os.makedirs(out_path + '/' + doc)
        axml.write(out_path + '/' + doc + '/' + xmlfile, pretty_print=True)
