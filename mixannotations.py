#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:19:28 2017

@author: egoitz
"""
import os
from lxml import etree
import re

props = ["AMPM-Of-Day", "End-Interval", "Interval", "Intervals", "Modifier",
         "Number", "Period", "Periods", "Repeating-Interval", "Repeating-Intervals",
         "Start-Interval", "Sub-Interval", "Time-Zone"]

cond = "[" + " or ".join(["self::%s" % p for p in props]) + "]"


#def getdata(path, tmlpath, rawpath):
#    for doc in os.listdir(path):
#        for xmlfile in os.listdir(path + '/' + doc):
#            axml = etree.parse(path + '/' + doc + '/' + xmlfile)
#            tmlxml = etree.parse(tmlpath + '/' + doc + '/' + doc + '.TimeML.gold.completed.xml')
#            rawfile = open(rawpath + '/' + doc, 'r')
#            text = rawfile.read()
#            rawfile.close()
#            entities = dict()
#            links = dict()
#            for entity in axml.findall('.//entity'):
#                eid = entity.find('./id').text
#                estart,eend = map(int,entity.find('./span').text.split(','))
#                etype = entity.find('./type').text
#                eparentsType = entity.find('./parentsType').text
#                entities[eid] = (estart,eend,etype,eparentsType,xmlfile,eid)
#                for prop in entity.xpath('./properties/*' + cond):
#                    ptag = prop.tag
#                    ptext = prop.text
#                    if ptext is not None:
#                        lids = axml.xpath('.//entity/id[text()="'+ ptext +'"]')
#                        if len(lids) == 1:
#                            lid = lids[0].text
#                            if eid not in links:
#                                links[eid] = list()
#                            links[eid].append(lid)
#            relations = dict()
#            for relation in tmlxml.findall('.//relation'):
#                rid = relation.find('./id').text
#                rtype = relation.find('./type').text
#                if rtype == "TLINK":
#                    rsource = relation.xpath('./properties/*')[0].text
#                    rtype = relation.xpath('./properties/*')[1].text
#                    rtarget = relation.xpath('./properties/*')[2].text
#                    if rsource not in relations:
#                        relations[rsource] = list()
#                    relations[rsource].append(rtype + ' ' + rtarget)
#                    if rtarget not in relations:
#                        relations[rtarget] = list()
#                    relations[rtarget].append('inv' + rtype + ' ' + rsource)                    
#            for entity in tmlxml.findall('.//entity'):
#                eid = entity.find('./id').text
#                estart,eend = map(int,entity.find('./span').text.split(','))
#                etype = entity.find('./type').text
#                for e in entities:
#                    start = entities[e][0]
#                    end = entities[e][1]
#                    if estart < end and start < eend:
#                        if eid not in relations:
#                            print (e,start,end,"".join(text[start:end]), eid,estart,eend,"".join(text[estart:eend]),"no-rel")
#                            #print (e,start,end,eid,estart,eend,"no-rel")
#                        else:
#                            for relation in relations[eid]:
#                                print (e,start,end,"".join(text[start:end]),eid,estart,eend,"".join(text[estart:eend]),relation)
#                                #print (e,start,end,eid,estart,eend,relation)
#                        if e in links:
#                            print ("\t" + " ".join(links[e]))
                    
                
def getdata(path, tmlpath, rawpath):
    for doc in os.listdir(path):
        for xmlfile in os.listdir(path + '/' + doc):
            axml = etree.parse(path + '/' + doc + '/' + xmlfile)
            tmlxml = etree.parse(tmlpath + '/' + doc + '/' + doc + '.TimeML.gold.completed.xml')
            rawfile = open(rawpath + '/' + doc, 'r')
            text = rawfile.read()
            rawfile.close()
            entities = dict()
            for entity in tmlxml.findall('.//entity'):
                eid = entity.find('./id').text
                estart,eend = map(int,entity.find('./span').text.split(','))
                etype = entity.find('./type').text
                if etype == "TIMEX3":
                    entities[eid] = (estart,eend,etype,xmlfile,eid,False)       
            for entity in axml.findall('.//entity'):
                eid = entity.find('./id').text
                estart,eend = map(int,entity.find('./span').text.split(','))
                etype = entity.find('./type').text
                both = False
                for e in entities:
                    start = entities[e][0]
                    end = entities[e][1]
                    if estart < end and start < eend:
                        both = True
                        print (e,start,end,re.sub(' ','_',"".join(text[start:end])), eid,estart,eend,re.sub(' ','_',"".join(text[estart:eend])))
                        ents = list(entities[e])
                        ents[-1] = True
                        entities[e] = tuple(ents)
                if not both:
                        print ("NotInTML", eid, estart, eend,re.sub(' ','_',"".join(text[estart:eend])))
            
            for e in entities:
                if not entities[e][-1]:
                    start = entities[e][0]
                    end = entities[e][1]
                    print (e,start,end,"".join(text[start:end]), "NotinTN")
                        
# train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
# test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
# train_tmlpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TempEval-2013-Train/TimeBank/'
# test_tmlpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TempEval-2013-Train/AQUAINT/'
# rawpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'

train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_tmlpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TempEval-2013-Train/TimeBank/'
test_tmlpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TempEval-2013-Train/AQUAINT/'
rawpath = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'


getdata(train_path, train_tmlpath, rawpath)
#getdata(test_path, test_tmlpath, rawpath)