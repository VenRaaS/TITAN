import os
import sys
import argparse
import json
from multiprocessing import Pool, Manager

import numpy as np


imgBN2Fea1D_dic = Manager().dict()
def compact_feas_dir(path) :
    imgFea_list = []
    imgBN_list = []

    print '{} is processing...'.format(path)
    for fn in os.listdir(path):
        if fn.endswith('.fea.npy'):
            fp = os.path.join(path, fn)
#            print fp
            imgFea = np.load(fp)
            imgBasename = fn.split('.')[0]
            imgBN2Fea1D_dic[imgBasename] = imgFea


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="the root dir of all image features")
    args = parser.parse_args()
   
    modelDir = os.path.normpath(args.dir) 
    if not os.path.isdir(modelDir):
        print 'an invalid dir: {}'.format(modelDir)
        sys.exit(1)

    paths = [ root for root, dirs, files in os.walk(modelDir) ]

    pool = Pool(processes=100)
    pool.map(compact_feas_dir, paths)

    print len(imgBN2Fea1D_dic)

    fea_na = np.asarray(imgBN2Fea1D_dic.values())
    bn_na = np.asarray(imgBN2Fea1D_dic.keys())

    if 0 < fea_na.size and 0 < bn_na.size:
        print os.path.join(modelDir, modelDir + '.feas')
        np.save(os.path.join(modelDir, modelDir + '.feas'), fea_na)

        print os.path.join(modelDir, modelDir + '.bns')
        np.save(os.path.join(modelDir, modelDir + '.bns'), bn_na)

