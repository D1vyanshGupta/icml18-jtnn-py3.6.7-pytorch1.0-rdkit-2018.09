import os
import sys
import pickle
from optparse import OptionParser
from multiprocessing import Pool

import rdkit

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

from jtnn import *

def tensorize(smiles, assm=True):
    junc_tree = MolJuncTree(smiles)
    junc_tree.recover()
    if assm:
        junc_tree.assemble()
        for node in junc_tree.nodes:
            if node.label not in node.candidates:
                node.candidates.append(node.label)

    # conserving memory
    del junc_tree.mol
    for node in junc_tree.nodes:
        del node.mol

    print('Done')
    return junc_tree

# def tensorize_pair(smiles_pair):
#     junc_tree0 = tensorize(smiles_pair[0], assm=False)
#     junc_tree1 = tensorize(smiles_pair[1], assm=True)
#     return (junc_tree0, junc_tree1)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    print('Before Pooling')
    all_data = pool.map(tensorize, data)
    print('After Pooling')

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        print("Split: {} created".format(split_id))
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)