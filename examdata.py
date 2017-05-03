#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:50:43 2017

@author: egoitz
"""

from lxml import etree
import os
import sys
import re
#sys.path.append('/home/egoitz/Tools/time/anaforatools')
#import anafora

#adata = anafora.AnaforaData()
#afile = adata.from_file('/home/egoitz/Desktop/prueba/gold/APW19980807.0261.TimeNorm.gold.completed.xml')
#axml = afile.xml

rawpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'
train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
path = train_path

props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
         "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
         "Start-Interval", "Sub-Interval", "Time-Zone"]

cond = " or ".join(["self::%s" % p for p in props])


#for doc in os.listdir(path):
#    for xmlfile in os.listdir(path + '/' + doc):
#        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
#        for entity in axml.findall('.//entity'):
#            eid = entity.find('./id').text
#            etype = entity.find('./type').text
#            eparensType = entity.find('./parentsType').text
#            props = []
#            for prop in entity.xpath('./properties/*'):
#                props.append(prop.tag)
#            print eid, etype, eparensType, " ".join(props)


#### print interval info
#for doc in os.listdir(path):
#    for xmlfile in os.listdir(path + '/' + doc):
#        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
#        rawfile = open(rawpath + '/' + doc, 'r')
#        text = rawfile.read()
#        rawfile.close()
#        for entity in axml.findall('.//entity'):
#            eid = entity.find('./id').text
#            etype = entity.find('./type').text
#            eparentsType = entity.find('./parentsType').text
#            estart, eend = map(int,entity.find('./span').text.split(','))
#            for prop in entity.xpath('./properties/Interval-Type'):
#                print (eid,etype,eparentsType,prop.text,re.sub(' ', '_', "".join(text[estart:eend])))


#### print property types
#for doc in os.listdir(path):
#    for xmlfile in os.listdir(path + '/' + doc):
#        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
#        rawfile = open(rawpath + '/' + doc, 'r')
#        text = rawfile.read()
#        rawfile.close()
#        for entity in axml.findall('.//entity'):
#            eid = entity.find('./id').text
#            etype = entity.find('./type').text
#            eparentsType = entity.find('./parentsType').text
#            estart, eend = map(int,entity.find('./span').text.split(','))
#            for ptype in entity.xpath('./properties/Type'):
#                print (eid,etype,eparentsType,ptype.text,re.sub(' ', '_', "".join(text[estart:eend])))
                        
### print links with entity pairs                   
for doc in os.listdir(path):
    for xmlfile in os.listdir(path + '/' + doc):
        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            etype = entity.find('./type').text
            eparensType = entity.find('./parentsType').text
            estart, eend = entity.find('./span').text.split(',')
            for prop in entity.xpath('./properties/*[' + cond + ']'):
                ptag = prop.tag
                pid = prop.text
                if pid is not None:
                    pentity = axml.find('.//entity[id="' + pid + '"]')
                    if pentity is not None:
                        ptype = pentity.find('./type').text
                        pparentsType = pentity.find('./parentsType').text
                        pstart, pend = pentity.find('./span').text.split(',')
                        print (eid, etype, eparensType, estart, eend, ptag, pid, ptype, pparentsType, pstart, pend, )
                    #else:
                    #    print "ERROR" + pid

#def getlinks(link, node, xml, cond):
#    for prop in node.xpath('./properties/*[' + cond + ']'):
#        #ptag = prop.tag
#        pid = prop.text
#        if pid is not None:
#            pentity = xml.find('.//entity[id="' + pid + '"]')
#            if pentity is not None:
#                #ptype = pentity.find('./type').text
#                #pparentsType = pentity.find('./parentsType').text
#                pid = pid + ":" + pentity.find('./span').text.split(',')[0]
#                if pid not in link:
#                    link.append(pid)
#                    link = getlinks(link, pentity, xml, cond)
#    return link
#
#links = []
#for doc in os.listdir(path):
#    for xmlfile in os.listdir(path + '/' + doc):
#        axml = etree.parse(path + '/' + doc + '/' + xmlfile)
#        for entity in axml.findall('.//entity'):
#            eid = entity.find('./id').text
#            eid = eid + ":" + entity.find('./span').text.split(',')[0]
#            link = [eid]
#            visited = False
#            for l in links:
#                if eid in l:
#                    visited = True
#            if not visited:
#                #etype = entity.find('./type').text
#                #eparensType = entity.find('./parentsType').text
#                link = getlinks(link, entity, axml, cond)
#                nlinks = []
#                for l in links:
#                    updated = False
#                    for nl in link:
#                        if nl in l:
#                            updated = True
#                    if not updated:
#                        nlinks.append(l)
#                links = nlinks
#                links.append(link)
#
#
#for link in links:
#    print " ".join(link)
#                    #else:
#                    #    print "ERROR" + pid
