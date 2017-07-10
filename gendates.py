import sys
import os
from lxml import etree
import dateutil.parser as dprs

rawpath = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/raw/'
train_path = '/home/egoitz//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'


def getexpression(entity, entities, expression):
    expression.append((entity,entities[entity][1],entities[entity][2]))
    for eproperty in entities[entity][5]:
        value = entities[entity][5][eproperty]
        if value in entities:
            expression = getexpression(value,entities,expression)
    return expression

for doc in os.listdir(train_path):
    for xmlfile in os.listdir(train_path + '/' + doc):
        axml = etree.parse(train_path + '/' + doc + '/' + xmlfile)
        rawfile = open(rawpath + '/' + doc, 'r')
        text = rawfile.read()
        rawfile.close()

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
            eproperties = dict()
            for eproperty in entity.findall('./properties/*'):
                eproperties[eproperty.tag] = eproperty.text
            if estart not in starts:
                starts[estart] = list()
            ent_values = (eid, estart, eend, etype, eparentsType, eproperties)
            starts[estart].append(eid)
            entities[eid] = ent_values
            

        firststart = sorted(starts)[0]
        entity = starts[firststart][0]
        expression = list()
        expression = getexpression(entity,entities,expression)
        expression.sort(key=lambda x: x[1])
        expstart = expression[0][1]
        expend = expression[-1][2]
        expspand = "".join(text[expstart:expend])
        date = dprs.parse(expspand)
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
            
        et = etree.ElementTree(root)
        et.write(sys.argv[1] + '/' + doc + '.auto.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")
        textfile = open(sys.argv[1] + '/' + doc,'w')
        textfile.write(newdate + '\n')
        textfile.close()