import sys
import json
from lxml import etree
from numpy import random

types = dict()
types['Month'] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
types['Day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def typekey(types, query):
    answ = None
    for key, values in types.items():
        if query in values:
            answ = key
    return answ
            
def read_from_json(filename):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

data = read_from_json(sys.argv[1])

for i in range(0, len(data)):
    doc = "rand" + str(i)
    root = etree.Element("data")
    anno = etree.SubElement(root,"annotations")
    e = 0
    diff = 0
    sent = (data[i][0])
    old = sent
    changed = False
    for o in data[i][1]:
        span_s = o[0] + diff
        span_e = o[1] + diff
        value = o[2]
        length = span_e - span_s
        key = typekey(types, value)
        if key is not None:
            changed = True
            new = value
            while new == value:
                new = random.choice(types[key], 1)[0]
            nlength = len(new)
            nsent = sent[:span_s]
            nsent = nsent + new
            nsent = nsent + sent[span_e:]
            sent = nsent
            diff = diff + (nlength - length)
            span_e = span_s + nlength
        for i in range(3, len(o)):
            type = o[i]
            ent = etree.SubElement(anno,"entity")
            eid = etree.SubElement(ent,"eid")
            eid.text = str(e) + "@" + doc + "@auto"
            span = etree.SubElement(ent,"span")
            span.text = str(span_s) + "," + str(span_e)
            etype = etree.SubElement(ent,"type")
            etype.text = type
            e += 1

    if changed:
        print('\n' + old)
        et = etree.ElementTree(root)
        print(etree.tostring(root, pretty_print=True))
        #et.write(sys.argv[2] + '/' + doc + '/' + doc + '.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")
        print(sent)
