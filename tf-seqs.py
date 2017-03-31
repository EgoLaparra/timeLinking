import getseqs
import sys

#train_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
#test_path = '/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
train_path = '/Users/laparra//Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/train_TimeBank/'
test_path = '/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/TimeNorm/test_AQUAINT/'
out_path = 'out/test/'


if sys.argv[1] == "t":
    links, entities, sequences,  max_seq = getseqs.getdata(train_path)
    (entitylists, transitions, newlinks,
     max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)

    in_vocab = set()
    out_vocab = set()
    f = open(sys.argv[2] + '/train-input','w')
    for seq in sequences:
        seqents = [entities[e][1] + ':' + entities[e][2] for e in sequences[seq]]
        for s in seqents:
            in_vocab.add(s)
        f.write(" ".join(seqents) + "\n")
    f.close()
        
    f = open(sys.argv[2] + '/train-input-vocab','w')
    f.write('\n'.join(in_vocab) + '\n')
    f.close()
    
    f = open(sys.argv[2] + '/train-output','w')
    for tran in transitions:
        for t in tran:
            out_vocab.add(t)
        f.write(" ".join(tran) + "\n")
    f.close()

    f = open(sys.argv[2] + '/train-output-vocab','w')
    f.write('\n'.join(out_vocab) + '\n')
    f.close()
    
    links, entities, sequences,  max_seq = getseqs.getdata(test_path)
    (entitylists, transitions, newlinks,
     max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)
    
    f = open(sys.argv[2] + '/test-input','w')
    m = open(sys.argv[2] + '/map','w')
    for seq in sequences:
        m.write(" ".join(sequences[seq]) + "\n")
        f.write(" ".join(entities[e][1] + ':' + entities[e][2] for e in sequences[seq]) + "\n")
    m.close()
    f.close()
        
    f = open(sys.argv[2] + '/test-output','w')
    for tran in transitions:
        f.write(" ".join(tran) + "\n")
    f.close()
        
elif sys.argv[1] == "p":
    links, entities, sequences,  max_seq = getseqs.getdata(test_path)
    (entitylists, transitions, newlinks,
     max_trans,transOp, trans2idx) = getseqs.get_transitions(links, entities, sequences)

    transitions = list()
    tfs2spredictions = open(sys.argv[2], 'r')
    for line in tfs2spredictions.readlines():
        seq = line.rstrip().split(' ')
        transitions.append(seq)
    tfs2spredictions.close()
    
    entitylists = list()
    m = open('seqs-dataset/map')
    for line in m:
        ments = line.rstrip().split(' ')
        entitylist = list()
        for e in ments:
            begin = entities[e][0].split(',')[0]
            entitylist.append((e, int(begin)))
        entitylist = sorted(entitylist, key=lambda x: x[1])
        entitylists.append(entitylist)
    m.close()
    
    outputs = getseqs.build_graph(entitylists, entities, transitions)
    
    getseqs.print_outputs(test_path,out_path, outputs)