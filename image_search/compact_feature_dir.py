import os
import sys
import argparse
import json
from multiprocessing import Pool

import numpy as np



def compact_feas_dir(path) :
    imgFea_list = []
    imgBN_list = []

    for fn in os.listdir(path):
        if fn.endswith('.fea.npy'):
            fp = os.path.join(path, fn)
#            print fp
            imgFea = np.load(fp)
            imgFea_list.append( imgFea )
            imgBN_list.append( fn.split('.')[0] )

    fea_na = np.asarray(imgFea_list)
    bn_na = np.asarray(imgBN_list)

    print '{} is processing...'.format(path)
    if 0 < fea_na.size and 0 < bn_na.size:
        dirname = os.path.split(path)[-1]
        np.save(os.path.join(path, dirname + '.feas'), fea_na)
        np.save(os.path.join(path, dirname + '.bns'), bn_na)


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="the root dir of all image features")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print 'an invalid dir: {}'.format(args.dir)
        sys.exit(1)

    paths = [ root for root, dirs, files in os.walk(args.dir) ]
    pool = Pool(processes=100)
    pool.map(compact_feas_dir, paths)

 
