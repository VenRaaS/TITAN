import os
import sys
import argparse
import json
from multiprocessing import Pool, Manager

import numpy as np


tImgBN2Fea1D_list = Manager().list()
def compact_feas_dir(fpath) :
    imgFea_list = []
    imgBN_list = []

    if fpath.endswith('.fea.npy'):
        imgFea = np.load(fpath)
        imgBasename = os.path.split(fpath)[-1].split('.')[0]
        tImgBN2Fea1D_list.append( (imgBasename, imgFea) )
        print 'bulk vector <= ({}, {})'.format(imgBasename, fpath)


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="the root dir of all image features")
    args = parser.parse_args()
   
    modelDir = os.path.normpath(args.dir) 
    if not os.path.isdir(modelDir):
        print 'an invalid dir: {}'.format(modelDir)
        sys.exit(1)

#    paths = [ root for root, dirs, files in os.walk(modelDir) ]
    files = [ os.path.join(modelDir, f) for f in os.listdir(modelDir) ]
    num_files = len(files)
    print 'total feature files: {}'.format(num_files)

    pool = Pool(processes=100)

    step = 50000
###    step = step if step <= num_files else num_files/2
    bi = 0
    for i in xrange(0, num_files, step):
        del tImgBN2Fea1D_list[:]
        pool.map(compact_feas_dir, files[i : i+step])

        basenames, feas = zip(*tImgBN2Fea1D_list)
        bn_na = np.asarray(basenames)
        fea_na = np.asarray(feas)

        if 0 < fea_na.size and 0 < bn_na.size:
            print os.path.join(modelDir, modelDir + '.{}.feas'.format(bi))
            np.save(os.path.join(modelDir, modelDir + '.{}.feas'.format(bi)), fea_na)

            print os.path.join(modelDir, modelDir + '.{}.bns'.format(bi))
            np.save(os.path.join(modelDir, modelDir + '.{}.bns'.format(bi)), bn_na)
       
        bi += 1



