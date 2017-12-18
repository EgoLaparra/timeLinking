import os
import sys
from lxml import etree
import json
from termcolor import colored
import numpy as np

import anafora
tnschema = anafora.get_schema('/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/.schema/timenorm-schema.xml')

np.random.seed(12345)

vocab = list(['PADD', 'UNK', 'EVENT', 'TIMEX'])
dep_inventory = list(['PADD', 'UNK'])

enomfile = open('event_nominalizations.txt', 'r')
enom = list()
for line in enomfile:
    v, n = line.rstrip().split()
    enom.append(n)
enomfile.close()


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
    t_deps = list()
    t_deps = get_deps(sent, t_id, '', t_deps)
    path = None
    for event in events:
        e_deps = list()
        e_deps = get_deps(sent, event, '', e_deps)
        path = get_path(t_deps, e_deps)
    sent_string.append(colored('-'.join(path), 'green'))
    return " ".join(sent_string)


def get_deps(sent, t_id, arc, deplist):
    deplist.append((t_id, arc))
    for dep in sent['basicDependencies']:
        if dep['dependent'] == t_id and not any(dep['governor'] == d[0] for d in deplist):
            deplist = get_deps(sent, dep['governor'], dep['dep'], deplist)
    return deplist


def get_path(deps_t, deps_e):
    path = list()
    r_deps_t = list(reversed(deps_t))
    r_deps_e = list(reversed(deps_e))
    c_node = None
    for d in range(0, len(r_deps_t)):
        if d == len(r_deps_e) or r_deps_t[d][0] != r_deps_e[d][0]:
            break
        else:
            c_node = r_deps_t[d][0]
    in_path = True
    for d in deps_t:
        if in_path and d[1] != '':
            path.append(d[1])
        if d[0] == c_node:
            in_path = False
    for d in r_deps_e:
        if d[0] == c_node:
            in_path = True
        if in_path and d[1] != '':
            path.append(d[1])
    return path


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
    t_deps = list()
    t_deps = get_deps(sent, t_id, '', t_deps)
    e_deps = list()
    e_deps = get_deps(sent, e_id, '', e_deps)
    path = None
    path = get_path(t_deps, e_deps)
    path_seq = list()
    for dep in path:
        if dep not in dep_inventory:
            dep_inventory.append(dep)
        didx = dep_inventory.index(dep)
        path_seq.append(didx)

    return (event, timex, sent_seq, path_seq)


def get_vocab_instance(cnlp, s_id, t_id, e_id, vocab, dep_inv=None):
    sent = cnlp['sentences'][s_id]
    event = None
    timex = None
    sent_seq = list()
    for token in sent['tokens']:
        if token['word'] not in vocab:
            vidx = 1  # UNK
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
    t_deps = list()
    t_deps = get_deps(sent, t_id, '', t_deps)
    e_deps = list()
    e_deps = get_deps(sent, e_id, '', e_deps)
    path = None
    path = get_path(t_deps, e_deps)
    path_seq = list()
    for dep in path:
        if dep_inv is not None:
            if dep not in dep_inv:
                didx = 1  # UNK
            else:
                didx = dep_inv.index(dep)
            path_seq.append(didx)
        else:
            if dep not in dep_inventory:
                dep_inventory.append(dep)
            didx = dep_inventory.index(dep)
            path_seq.append(didx)

    return (event, timex, sent_seq, path_seq)


def get_embs(embfile):
    vocab = list(['PADD', 'UNK', 'EVENT', 'TIMEX'])
    embs = list([[0.0]*200 for i in range(0, 4)])
    with open(embfile, 'r') as efile:
        for line in efile:
            fields = line.rstrip().split(' ')
            w = fields[0]
            e = list(map(float, fields[1:]))
            vocab.append(w)
            embs.append(e)
    return vocab, embs


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


def shuffle(x, y):
    idx = list(range(0, len(x)))
    np.random.shuffle(idx)
    s_x = list()
    s_y = list()
    for i in idx:
        s_x.append(x[i])
        s_y.append(y[i])
    return s_x, s_y


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


def closest_events(token, events):
    if len(events) > 0:
        events = np.array(events)
        dist = np.abs(events - token)
        min_dist = np.min(dist)
        closests = np.reshape(np.argwhere(dist == min_dist), -1)
        close_events = events[closests]
        return list(close_events)
    else:
        return events


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
                    # events = closest_events(token, events)
                    for event in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
                        linked = islinked(axml, entity, event_start, event_lemma)
                        instance = get_instance(cnlp, sent, token, event)
                        if is_candidate(cnlp, sent, token, [event], etype):
                            if instance[0] is not None and instance[1] is not None:
                                # path = get_path(cnlp, sent, token, event)
                                data_x.append(instance)
                                if linked:
                                    data_y.append(1)
                                else:
                                    data_y.append(0)
    return data_x, data_y, vocab, dep_inventory


def get_vocab_data(cnl, anafora, vocab, dep_inv=None):
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
                    # events = closest_events(token, events)
                    for event in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
                        linked = islinked(axml, entity, event_start, event_lemma)
                        instance = get_vocab_instance(cnlp, sent, token, event, vocab, dep_inv=dep_inv)
                        #if (dep_inv is None and linked) or is_candidate(cnlp, sent, token, [event], etype):
                        if is_candidate(cnlp, sent, token, [event], etype):
                            if instance[0] is not None and instance[1] is not None:
                                data_x.append(instance)
                                if linked:
                                    data_y.append(1)
                                else:
                                    data_y.append(0)
    return data_x, data_y, dep_inventory


# def is_candidate(cnlp, s_id, t_id, events, etype):
#     sent = cnlp['sentences'][s_id]
#     sent_string = list()
#     t = False
#     e = False
#     q = False
#     s = False
#     t_token= ""
#     for token in sent['tokens']:
#         if token['index'] == t_id:
#             if e and q:
#                 s = True
#             t = True
#             t_token = token['word'] + ' ' + etype
#             sent_string.append(colored(token['word'] + etype, 'red'))
#         elif token['index'] in events:
#             if token['lemma'] in ["say", "tell", "ask", "argue"]:
#                 if t and q:
#                     s = True
#                 e = True
#             t_token = t_token + ' ' + token['word']
#             sent_string.append(colored(token['word'], 'blue'))
#         else:
#             if (t or e) and (token['word'] == "''" or token['word'] == "``"):
#                 if q:
#                     q = False
#                 else:
#                     q = True
#             sent_string.append(token['word'])
#     t_deps = list()
#     t_deps = get_deps(sent, t_id, '', t_deps)
#     path = None
#     child = False
#     for ev in events:
#         e_deps = list()
#         e_deps = get_deps(sent, ev, '', e_deps)
#         path = get_path(t_deps, e_deps)
#         coma = False
#         for i in range(min(ev, t_id), max(ev, t_id)):
#             if sent['tokens'][i-1]['word'] in [",", "and", "or"]:
#                 coma = True
#         if not coma and (etype == "Before" or etype == "After" or etype == "Between"):
#             s = True
        

#     return s

def is_candidate(cnlp, s_id, t_id, events, etype):
    sent = cnlp['sentences'][s_id]
    t_deps = list()
    t_deps = get_deps(sent, t_id, '', t_deps)
    path = None
    s = False
    for ev in events:
        e_deps = list()
        e_deps = get_deps(sent, ev, '', e_deps)
        path = get_path(t_deps, e_deps)
        if etype == "Before" or etype == "After" or etype == "Between":
            if len(path) > 0 and (path[0] == "mark" or path[0] == "case"):
                s = True
        if (etype == "Last" or etype == "Next" or etype == "This"):
            roots = [d in path for d in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl', 'ccomp', 'xcomp']]
            if np.sum(roots) == 1:
        #    if e_deps[1][1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
                if path[-1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl'] or path[0] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
                    s = True
    return s


if __name__ == "__main__":
    for file in os.listdir(sys.argv[1]):
        cnlppath = os.path.join(sys.argv[2], file + '.json')
        xmlpath = os.path.join(sys.argv[1], file, file + '.TimeNorm.gold.completed.xml')
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
                    # events = closest_events(token, events)
                    for event in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][event - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][event - 1]['characterOffsetBegin']
                        linked = islinked(axml, entity, event_start, event_lemma)
                        string_events = string_sent_events(cnlp, sent, token, [event], etype)
                        print("%s\t%s" % (linked, string_events))
