import sys
import os
from lxml import etree
import dateutil.parser as dprs
import datetime
from numpy import random

        
for d in random.randint(0,high=1600000000,size=int(sys.argv[1])):
    date = datetime.datetime.fromtimestamp(d)
    docdate = date.strftime('%Y%m%d')
    doc = "".join(["randdate",docdate])

    newdate = date.strftime('%Y-%m-%d')
    spans = list()
    start = 0
    for c in range(0,len(newdate)):
        if newdate[c] == "-":
            end = c
            spans.append((str(start),str(end)))
            start = c + 1
    end = len(newdate)
    spans.append((str(start),str(end)))
    
    newvalues = date.strftime('%Y-%B-%d')
    values = newvalues.split('-')

    
    values = newvalues.split('-')
    types = list()
    types.append("Year")
    types.append("Month-Of-Year")
    types.append("Day-Of-Month")
    ptypes = list()
    ptypes.append("Interval")
    ptypes.append("Repeating-Interval")
    ptypes.append("Repeating-Interval")
    
    root = etree.Element("data")
    anno = etree.SubElement(root,"annotation")
    for e,(s,v,t,p) in enumerate(zip(spans,values,types,ptypes)):
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
            subint.text = str(e+1) + "@" + doc + "@auto"
            modifier = etree.SubElement(properties,"Modifier")
        elif t == "Month-Of-Year":
            value = etree.SubElement(properties,"Type")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            subint.text = str(e+1) + "@" + doc + "@auto"
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
        else:
            value = etree.SubElement(properties,"Value")
            value.text = v
            subint = etree.SubElement(properties,"Sub-Interval")
            number = etree.SubElement(properties,"Number")
            modifier = etree.SubElement(properties,"Modifier")
    
    if not os.path.exists(sys.argv[2] + '/' + doc):
        os.mkdir(sys.argv[2] + '/' + doc)
    et = etree.ElementTree(root)
    et.write(sys.argv[2] + '/' + doc + '/' + doc + '.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")
    textfile = open(sys.argv[2] + '/' + doc + '/' + doc,'w')
    textfile.write(newdate + '\n')
    textfile.close()