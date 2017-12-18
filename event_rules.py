import event

from lxml import etree
import json
import os
import sys
import math
import numpy as np
from termcolor import colored

import anafora
tnschema = anafora.get_schema('/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/.schema/timenorm-schema.xml')

enomfile = open('event_nominalizations.txt', 'r')
enom = list()
for line in enomfile:
    v, n = line.rstrip().split()
    enom.append(n)
enomfile.close()


def is_candidate(cnlp, s_id, t_id, events, etype):
    sent = cnlp['sentences'][s_id]
    sent_string = list()
    t = False
    e = False
    q = False
    s = False
    for token in sent['tokens']:
        if token['index'] == t_id:
            if e and q:
                s = True
            t = True
            sent_string.append(colored(token['word'] + etype, 'red'))
        elif token['index'] in events:
            if token['lemma'] in ["say", "tell", "ask", "argue"]: # 'xcomp', 'ccomp', 
                if t and q:
                    s = True
                e = True
        else:
            if (t or e) and (token['word'] == "''" or token['word'] == "``"):
                if q:
                    q = False
                else:
                    q = True

    t_deps = list()
    t_deps = event.get_deps(sent, t_id, '', t_deps)
    path = None
    s = False
    for ev in events:
        e_deps = list()
        e_deps = event.get_deps(sent, ev, '', e_deps)
        path = event.get_path(t_deps, e_deps)
        
        if (etype == "Before" or etype == "After" or etype == "Between"):
            if len(path) > 0 and (path[0] == "mark" or path[0] == "case"):
                s = True
                
        if (etype == "Last" or etype == "Next" or etype == "This"):
            roots = [d in path for d in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl', 'ccomp', 'xcomp']]
            if np.sum(roots) == 1:
        #    if e_deps[1][1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
                if path[-1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl'] or path[0] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
                    s = True

    sent_string.append(colored('-'.join(path), 'green'))
    if len(path) > 0:
        sent_string.append(path[0])
        sent_string.append(path[-1])
    # sent_string.append(colored(str(s), 'yellow'))
    sent_string.append(str(s))
    # sent_string.append(str(child))
    # sent_string = [t_token]
    return " ".join(sent_string)
 


def string_sent_events(cnlp, s_id, t_id, events, etype):
    sent = cnlp['sentences'][s_id]
    sent_string = list()
    t = False
    e = False
    q = False
    s = False
    t_token= ""
    for token in sent['tokens']:
        if token['index'] == t_id:
            if e and q:
                s = True
            t = True
            t_token = token['word'] + ' ' + etype
            sent_string.append(colored(token['word'] + etype, 'red'))
        elif token['index'] in events:
            if token['lemma'] in ["say", "tell", "ask", "argue"]: # 'xcomp', 'ccomp', 
                if t and q:
                    s = True
                e = True
            t_token = t_token + ' ' + token['word']
            sent_string.append(colored(token['word'], 'blue'))
        else:
            if (t or e) and (token['word'] == "''" or token['word'] == "``"):
                if q:
                    q = False
                else:
                    q = True
            sent_string.append(token['word'])
    # s = False
    t_deps = list()
    t_deps = event.get_deps(sent, t_id, '', t_deps)
    path = None
    child = False
    for ev in events:
        e_deps = list()
        e_deps = event.get_deps(sent, ev, '', e_deps)
        path = event.get_path(t_deps, e_deps)
        coma = False
        for i in range(min(ev, t_id), max(ev, t_id)):
            if sent['tokens'][i-1]['word'] in [",", "and", "or"]:
                coma = True
        if not coma and 5 > ev - t_id > 0 and (etype == "Before" or etype == "After" or etype == "Between") and sent['tokens'][ev-1]['lemma'] not in ["be", "have"]:
            s = True
            s = False
        if len(path) == 1 and (path[0] == "mark" or path[0] == "case"):
            s = True
        # s = False
        # if (etype == "Last" or etype == "Next" or etype == "This") and token['lemma'] not in ["say", "tell", "ask", "argue"]:
        #    roots = [d in path for d in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl', 'ccomp', 'xcomp']]
        #    if np.sum(roots) == 1:
            #    if e_deps[1][1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
        #        if path[-1] in ['acl', 'acl:relcl', 'conj', 'csub', 'advcl']:
        #            s = True
    sent_string.append(colored('-'.join(path), 'green'))
    if len(path) > 0:
        sent_string.append(path[0])
        sent_string.append(path[-1])
    # sent_string.append(colored(str(s), 'yellow'))
    sent_string.append(str(s))
    # sent_string.append(str(child))
    # sent_string = [t_token]
    return " ".join(sent_string)


def G(adir, jdir):
    for file in os.listdir(adir):
        cnlppath = os.path.join(jdir, file + '.json')
        xmlpath = os.path.join(adir, file, file + '.TimeNorm.gold.completed.xml')
        cnlp = json.load(open(cnlppath))
        axml = etree.parse(xmlpath)
        for entity in axml.findall('.//entity'):
            eid = entity.find('./id').text
            estart, eend = map(int, entity.find('./span').text.split(','))
            etype = entity.find('./type').text
            eparentType = entity.find('./parentsType').text
            if event.get_relation(tnschema, etype, 'Event') is not None:
                lentity, link = event.get_link(axml, eid)
                if lentity is None:
                    sent, token = event.get_token(cnlp, estart)
                    events = event.get_events(cnlp, sent, enom)
                    # events = closest_events(token, events)
                    for e in events:
                        event_lemma = cnlp['sentences'][sent]['tokens'][e - 1]['lemma']
                        event_start = cnlp['sentences'][sent]['tokens'][e - 1]['characterOffsetBegin']
                        linked = event.islinked(axml, entity, event_start, event_lemma)
                        # string_events = string_sent_events(cnlp, sent, token, [e], etype)
                        string_events = is_candidate(cnlp, sent, token, [e], etype)
                        print("%s\t%s" % (linked, string_events))


#embfile = 'embs.list'
#vocab, embs = event.get_embs(embfile)
cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train_corenlp/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train/train/'
# train_x, train_y, vocab, dep_inventory = event.get_data(cnlp, anafora)
#train_x, train_y, dep_inventory = event.get_vocab_data(cnlp, anafora, vocab)
# train_x, train_y = event.shuffle(train_x, train_y)
#cnlp = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_corenlp/'
#anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/test_gold/'
anafora = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/newres/train/dev/'
#test_x, test_y, _ = event.get_vocab_data(cnlp, anafora, vocab, dep_inv=dep_inventory)

G(anafora, cnlp)

#predicition = np.array(list(map(round, prediction)))
#test_y = np.array(test_y)

#print('Pred: %d - True: %d - Acc: %d' % (np.sum(predicition == 1), np.sum(test_y == 1), np.sum(predicition + test_y == 2)))
