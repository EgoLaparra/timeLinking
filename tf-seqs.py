import getseqs
import sys

train_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'

links, entities, sequences,  max_seq = getseqs.getdata(test_path)
(entitylists, transitions, newlinks,
 max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)

#entities, sequences,  _ = getseqs.get_testdata(test_path)
transitions = list()
tfs2spredictions = open(sys.argv[1], 'r')
for line in tfs2spredictions.readlines():
    seq = line.rstrip().split(' ')
    transitions.append(seq)
tfs2spredictions.close()

entitylists = list()
for key in sequences.keys():
    arcs = list()
    transition = list()
    entitylist = list()
    for entity in sequences[key]:
        begin = entities[entity][0].split(',')[0]
        entitylist.append((entity, int(begin)))
    entitylist = sorted(entitylist, key=lambda x: x[1])
    entitylists.append(entitylist)

outputs = getseqs.build_graph(entitylists, entities, transitions)

getseqs.print_outputs(test_path,out_path, outputs)