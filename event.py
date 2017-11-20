import os
import sys
from lxml import etree
import json
from termcolor import colored

import anafora
tnschema = anafora.get_schema('/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/.schema/timenorm-schema.xml')

vocab = list(['PADD', 'UNK', 'EVENT', 'TIMEX'])


def get_relation(tnschema, parent, child):
    if parent in tnschema:
        for relation in tnschema[parent]:
            if relation != "parentsType":
                for validChild in tnschema[parent][relation][1]:
                    if validChild == child:
                        return relation
    return None


def get_token(cnlp, start):
    for s in range(0, len(cnlp['sentences'])):
        sent = cnlp['sentences'][s]
        for token in sent['tokens']:
            if token["characterOffsetBegin"] <= start and token["characterOffsetEnd"] >= start:
                return s, token["index"]


def print_sent(cnlp, s_id, t_id, extra, color):
    sent = cnlp['sentences'][s_id]
    sent_string = list()
    for token in sent['tokens']:
        if token['index'] == t_id:
            sent_string.append(colored(token['word'] + extra, color))
        else:
            sent_string.append(token['word'])
    print(" ".join(sent_string))


def get_link(xml, id):
    for entity in xml.findall('.//entity'):
        for property in entity.findall('./properties/*'):
            if property.text == id:
                return entity, property.tag
    return None, None


def get_events(cnlp, s_id, enom):
    sent = cnlp['sentences'][s_id]
    events = list()
    for token in sent['tokens']:
        if token['pos'].startswith('VB') or token['lemma'] in enom:
            events.append(token['index'])
    return events


def string_sent_events(cnlp, s_id, t_id, events, etype):
    sent = cnlp['sentences'][s_id]
    sent_string = list()
    for token in sent['tokens']:
        if token['index'] == t_id:
            sent_string.append(colored(token['word'] + etype, 'red'))
        elif token['index'] in events:
            sent_string.append(colored(token['word'], 'blue'))
        else:
            sent_string.append(token['word'])
    return " ".join(sent_string)


def get_instance(cnlp, s_id, t_id, e_id):
    sent = cnlp['sentences'][s_id]
    event = None
    timex = None
    sent_seq = list()
    for token in sent['tokens']:
        if token['word'] not in vocab:
            vocab.append(token['word'])
        vidx = vocab.index(token['word'])
        if token['index'] == t_id:
            timex = vidx
            sent_seq.append(3)  # TIMEX
        elif token['index'] == e_id:
            event = vidx
            sent_seq.append(2)  # EVENT
        else:
            sent_seq.append(vidx)
    return (event, timex, sent_seq)


def get_test_instance(cnlp, s_id, t_id, e_id, vocab):
    sent = cnlp['sentences'][s_id]
    event = None
    timex = None
    sent_seq = list()
    for token in sent['tokens']:
        if token['word'] not in vocab:
            vidx = 1
        else:
            vidx = vocab.index(token['word'])
        if token['index'] == t_id:
            timex = vidx
            sent_seq.append(3)  # TIMEX
        elif token['index'] == e_id:
            event = vidx
            sent_seq.append(2)  # EVENT
        else:
            sent_seq.append(vidx)
    return (event, timex, sent_seq)


def islinked(axml, target, event_start, l):
    event_id = None
    for entity in axml.findall('.//entity'):
        eid = entity.find('./id').text
        estart, eend = map(int, entity.find('./span').text.split(','))
        etype = entity.find('./type').text
        if estart <= event_start and eend >= event_start and etype == "Event":
            event_id = eid

    if event_id is not None:
        for property in target.findall('./properties/*'):
            if property.text == event_id:
                return True

    return False


enomfile = open('event_nominalizations.txt', 'r')
enom = list()
for line in enomfile:
    v, n = line.rstrip().split()
    enom.append(n)
enomfile.close()


# for file in os.listdir(sys.argv[1]):
#     cnlppath = os.path.join(sys.argv[2], file + '.json')
#     xmlpath = os.path.join(sys.argv[1], file, file + '.TimeNorm.gold.completed.xml')
#     cnlp = json.load(open(cnlppath))
#     axml = etree.parse(xmlpath)
#     for entity in axml.findall('.//entity'):
#         eid = entity.find('./id').text
#         estart, eend = map(int, entity.find('./span').text.split(','))
#         etype = entity.find('./type').text
#         if etype == "Event":
#             sent, token = get_token(cnlp, estart)
#             lentity, link = get_link(axml, eid)
#             if lentity is not None:
#                 print(eid)
#                 print_sent(cnlp, sent, token, link, 'red')
#                 lestart, leend = map(int, lentity.find('./span').text.split(','))
#                 lsent, ltoken = get_token(cnlp, lestart)
#                 letype = lentity.find('./type').text
#                 print_sent(cnlp, lsent, ltoken, letype, 'blue')
#             else:
#                 print_sent(cnlp, sent, token, 'NONE', 'red')
#                 print("NONE")
#             print()

# for file in os.listdir(sys.argv[1]):
#     cnlppath = os.path.join(sys.argv[2], file + '.json')
#     xmlpath = os.path.join(sys.argv[1], file, file + '.TimeNorm.gold.completed.xml')
#     cnlp = json.load(open(cnlppath))
#     axml = etree.parse(xmlpath)
#     for entity in axml.findall('.//entity'):
#         eid = entity.find('./id').text
#         estart, eend = map(int, entity.find('./span').text.split(','))
#         etype = entity.find('./type').text
#         eparentType = entity.find('./parentsType').text
#         if get_relation(tnschema, etype, 'Event') is not None:
#             lentity, link = get_link(axml, eid)
#             if lentity is None:
#                 sent, token = get_token(cnlp, estart)
#                 events = get_events(cnlp, sent, enom)
#                 for event in events:
#                     event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
#                     event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
#                     linked = islinked(axml, entity, event_start, event_lemma)
#                     string_events = string_sent_events(cnlp, sent, token, [event], etype)
#                     print("%s\t%s" % (linked, string_events))

def get_data(cnl, anafora):
    data_x = list()
    data_y = list()
    for file in os.listdir(anafora):
        cnlppath = os.path.join(cnl, file + '.json')
        xmlpath = os.path.join(anafora, file, file + '.TimeNorm.gold.completed.xml')
        cnlp = json.load(open(cnlppath))
        axml = etree.parse(xmlpath)
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = map(int, entity.find('./span').text.split(','))
            etype = entity.find('./type').text
            eparentType = entity.find('./parentsType').text
            if get_relation(tnschema, etype, 'Event') is not None:
                lentity, link = get_link(axml, eid)
                if lentity is None:
                    sent, token = get_token(cnlp, estart)
                    events = get_events(cnlp, sent, enom)
                    for event in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
                        linked = islinked(axml, entity, event_start, event_lemma)
                        instance = get_instance(cnlp, sent, token, event)
                        if instance[0] is not None and instance[1] is not None:
                            data_x.append(instance)
                            if linked:
                                data_y.append(1)
                            else:
                                data_y.append(0)
    return data_x, data_y, vocab


def get_test_data(cnl, anafora, vocab):
    data_x = list()
    data_y = list()
    for file in os.listdir(anafora):
        cnlppath = os.path.join(cnl, file + '.json')
        xmlpath = os.path.join(anafora, file, file + '.TimeNorm.gold.completed.xml')
        cnlp = json.load(open(cnlppath))
        axml = etree.parse(xmlpath)
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = map(int, entity.find('./span').text.split(','))
            etype = entity.find('./type').text
            eparentType = entity.find('./parentsType').text
            if get_relation(tnschema, etype, 'Event') is not None:
                lentity, link = get_link(axml, eid)
                if lentity is None:
                    sent, token = get_token(cnlp, estart)
                    events = get_events(cnlp, sent, enom)
                    for event in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
                        linked = islinked(axml, entity, event_start, event_lemma)
                        instance = get_test_instance(cnlp, sent, token, event, vocab)
                        if instance[0] is not None and instance[1] is not None:
                            data_x.append(instance)
                            if linked:
                                data_y.append(1)
                            else:
                                data_y.append(0)
    return data_x, data_y
