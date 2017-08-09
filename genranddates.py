import sys
import os
from lxml import etree
import dateutil.parser as dprs
import datetime
import numpy as np
from numpy import random

if sys.platform == "linux" or sys.platform == "linux2":
    dtformats = [("%-m/%-d/%y", "%Y/%-d/%B"), #7/24/17
                 #("%A, %B %d, %Y", "%A-%B-%d-%Y"), #Monday, July 24, 2017
                 ("%m/%d/%y", "%B/%-d/%Y"), #07/24/17
                 ("%m/%d/%Y", "%B/%-d/%Y"), #07/24/2017
                 ("%b %-d, %y", "%B/%-d/%Y"), #Jul 24, 17
                 ("%b %-d, %Y", "%B/%-d/%Y"), #Jul 24, 2017
                 ("%-d. %-b. %Y", "%-d/%B/%Y"), #24. Jul. 2017
                 ("%B %-d, %Y", "%B/%-d/%Y"), #July 24, 2017
                 ("%-d. %B %Y", "%-d/%B/%Y"), #24. July 2017
                 #("%a, %b %-d, %y", "%A-%B-%d-%Y"), #Mon, Jul 24, 17
                 #("%a %d/%b %y", "%A-%d-%B-%Y"), #Mon 24/Jul 17
                 #("%a, %B %-d, %Y", "%A-%B-%d-%Y"), #Mon, July 24, 2017
                 #("%A, %B %-d, %Y", "%A-%B-%d-%Y"), #Monday, July 24, 2017
                 ("%m-%d", "%B/%-d"), #07-24
                 ("%y-%m-%d", "%Y/%B/%-d"), #17-07-24
                 ("%Y-%m-%d", "%Y/%B/%-d"), #2017-07-24
                 ("%m/%d", "%B/%-d"), #07/24
                 ("%b %d", "%B/%-d"), #Jul 24
                 ("%m/%d/%y %H:%M %p", "%B/%-d/%Y/%-H/%-M/%p"), #07/24/17 09:52 AM
                 ("%m/%d/%Y %H:%M:%S", "%B/%-d/%Y/%-H/%-M/%-S"), #07/24/2017 09:52:57
                 ("%Y-%m-%d %H:%M:%S", "%Y/%B/%-d/%-H/%-M/%-S"), #2017-07-24 09:52:57
                 ]
    
    tokord = {"%Y" : 0,
          "%B" : 1,
          "%-d" : 2,
          "%-H" : 3,
          "%-M" : 4,
          "%-S" : 5,
          "%p" : 6,
        } 
else:
    dtformats = [("%#m/%#d/%y", "%Y/%#d/%B"), #7/24/17
                 #("%A, %B %d, %Y", "%A-%B-%d-%Y"), #Monday, July 24, 2017
                 ("%m/%d/%y", "%B/%#d/%Y"), #07/24/17
                 ("%m/%d/%Y", "%B/%#d/%Y"), #07/24/2017
                 ("%b %#d, %y", "%B/%#d/%Y"), #Jul 24, 17
                 ("%b %#d, %Y", "%B/%#d/%Y"), #Jul 24, 2017
                 ("%#d. %b. %Y", "%#d/%B/%Y"), #24. Jul. 2017
                 ("%B %#d, %Y", "%B/%#d/%Y"), #July 24, 2017
                 ("%#d. %B %Y", "%#d/%B/%Y"), #24. July 2017
                 #("%a, %b %-d, %y", "%A-%B-%d-%Y"), #Mon, Jul 24, 17
                 #("%a %d/%b %y", "%A-%d-%B-%Y"), #Mon 24/Jul 17
                 #("%a, %B %-d, %Y", "%A-%B-%d-%Y"), #Mon, July 24, 2017
                 #("%A, %B %-d, %Y", "%A-%B-%d-%Y"), #Monday, July 24, 2017
                 ("%m-%d", "%B/%#d"), #07-24
                 ("%y-%m-%d", "%Y/%B/%#d"), #17-07-24
                 ("%Y-%m-%d", "%Y/%B/%#d"), #2017-07-24
                 ("%m/%d", "%B/%#d"), #07/24
                 ("%b %d", "%B/%#d"), #Jul 24
                 ("%m/%d/%y %H:%M %p", "%B/%#d/%Y/%#H/%#M/%p"), #07/24/17 09:52 AM
                 ("%m/%d/%Y %H:%M:%S", "%B/%#d/%Y/%#H/%#M/%#S"), #07/24/2017 09:52:57
                 ("%Y-%m-%d %H:%M:%S", "%Y/%B/%#d/%#H/%#M/%#S"), #2017-07-24 09:52:57
                 ]
    
    tokord = {"%Y" : 0,
          "%B" : 1,
          "%#d" : 2,
          "%#H" : 3,
          "%#M" : 4,
          "%#S" : 5,
          "%p" : 6,
        } 

    

types = [("Year", "Interval"),
         ("Month-Of-Year", "Repeating-Interval"),
         ("Day-Of-Month", "Repeating-Interval"),
         ("Hour-Of-Day", "Repeating-Interval"),
         ("Minute-Of-Hour", "Repeating-Interval"),
         ("Second-Of-Minute", "Repeating-Interval"),
         ("Second-Of-Minute", "Repeating-Interval")
         ]


for d in random.randint(0,high=1600000000,size=int(sys.argv[1])):
    date = datetime.datetime.fromtimestamp(d)
    docdate = date.strftime('%Y%m%d')
    doc = "".join(["randdate",docdate])

    f = random.randint(0,high=len(dtformats),size=1)[0]
    newdate = date.strftime(dtformats[f][0])
    spans = list()
    start = 0
    for c in range(0,len(newdate)):
        if newdate[c] == "-" or newdate[c] == "/" or newdate[c] == " " or newdate[c] == ":":
            end = c
            spans.append((str(start),str(end)))
            start = c + 1
    end = len(newdate)
    spans.append((str(start),str(end)))
    
    newvalues = date.strftime(dtformats[f][1])  
    values = newvalues.split('/')
    order = [tokord[s] for s in dtformats[f][1].split("/")]
    sort_order = np.argsort(order)
    
    root = etree.Element("data")
    anno = etree.SubElement(root,"annotations")
    for e,o in enumerate(sort_order):
        s = spans[o]
        v = values[o]
        (t, p) = types[order[o]]
        ent = etree.SubElement(anno,"entity")
        eid = etree.SubElement(ent,"eid")
        eid.text = str(e) + "@" + doc + "@auto"
        span = etree.SubElement(ent,"span")
        span.text = ",".join(s)
        etype = etree.SubElement(ent,"type")
        etype.text = t
        ptype = etree.SubElement(ent,"parentsType")
        ptype.text = p
        properties= etree.SubElement(ent,"properties")
        if t == "Year":
            value = etree.SubElement(properties,"Value")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            if e+1 < len(sort_order):
                subint.text = str(e+1) + "@" + doc + "@auto"
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Month-Of-Year":
            value = etree.SubElement(properties,"Type")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            if e+1 < len(sort_order):
                subint.text = str(e+1) + "@" + doc + "@auto"
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Day-Of-Month":
            value = etree.SubElement(properties,"Value")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            if e+1 < len(sort_order):
                subint.text = str(e+1) + "@" + doc + "@auto"
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Hour-Of-Day":
            value = etree.SubElement(properties,"Value")
            value.text = v
            ampm = etree.SubElement(properties,"AMPM-Of_Day")
            last = sort_order[-1]
            last = order[last]
            if types[last][0] == "AMPM-Of-Day":
                ampm.text = str(len(sort_order) - 1) + "@" + doc + "@auto"
            zone = etree.SubElement(properties,"Time-Zone")
            subint = etree.SubElement(properties,"Sub-Interval")
            if e+1 < len(sort_order):
                subint.text = str(e+1) + "@" + doc + "@auto"
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Minute-Of-Hour":
            value = etree.SubElement(properties,"Value")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            last = sort_order[-1]
            last = order[last]
            if types[last][0] == "Second-Of-Minute":
                subint.text = str(len(sort_order) - 1) + "@" + doc + "@auto"
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Second-Of-Minute":
            value = etree.SubElement(properties,"Value")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "AMPM-Of-Day":
            value = etree.SubElement(properties,"Value")
            value.text = v
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
    
    if not os.path.exists(sys.argv[2] + '/' + doc):
        os.mkdir(sys.argv[2] + '/' + doc)
    et = etree.ElementTree(root)
    et.write(sys.argv[2] + '/' + doc + '/' + doc + '.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")
    textfile = open(sys.argv[2] + '/' + doc + '/' + doc,'w')
    textfile.write(newdate + '\n')
    textfile.close()
    
    