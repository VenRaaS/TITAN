# coding: utf-8

import argparse
import os
import sys
import datetime
import shutil
import itertools
import shutil
import logging
from multiprocessing import Pool, Manager

from keras.preprocessing import image
import numpy as np
from skimage import io
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import keras.applications.vgg16 as vgg16
from keras.applications.vgg16 import decode_predictions

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")



tImg4DImgFN_list = Manager().list()
def data_preprocess(imgFP) :
    try:
        if imgFP.endswith('jpg'):
            img = image.load_img(imgFP, target_size=(224, 224))
            img3D = image.img_to_array(img)
            img4D = np.expand_dims(img3D, axis=0)
            imgFN = imgFP
            tImg4DImgFN_list.append( (img4D, imgFN) )
    except Exception as e:
        logging.error(str(e))


def cp_src2dst(imgFP, imgTrainDir) :
    if not os.path.exists(imgTrainDir):
        os.makedirs(imgTrainDir)

    logging.info('cp {} {}'.format(imgFP, imgTrainDir))
    shutil.copy(imgFP, imgTrainDir)


def model_vgg16_fc2() :
    model_base = vgg16.VGG16(weights='imagenet')
    model = Model(inputs=model_base.input, outputs=model_base.get_layer('fc2').output)
    return model


##
## return nearest neighbor, whose distance (between the target) <= threshold 
##
def nn_lte_threshold(img_feavct, imgFP, part_feavcts, part_imgFPs, threshold = 1.0e-3) :
    
    nn_imgFPs = []

    for part_i in range(len(part_imgFPs)):
        feavcts = part_feavcts[part_i]
        dists = np.square(img_feavct - feavcts).mean(axis=1)
        sort_ids = np.argsort(dists)
        #-- indices who makes the condition True
        nn_ids =  np.where(dists <= threshold)[0]

        filepaths = [ part_imgFPs[part_i][i] for i in nn_ids ]
        #-- exclude self
        filepaths = [ p for p in filepaths if p != imgFP ]

        if 0 < len(filepaths):
            nn_imgFPs += filepaths 
            logging.info('nestnear neighbors (<={}) of {}: {}'.format(threshold, imgFP, filepaths))

    return nn_imgFPs 


if '__main__' == __name__ :
    parser = argparse.ArgumentParser()
    parser.add_argument("dirImgs", help="the directory of the source images")
    parser.add_argument("dirResult", help="the root directory for a dir per file result")
    parser.add_argument("--nnt", default=1.0e-3, help="the threshold of nearest neighbor")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dirImgs):
        logging.error('an invalid dir: {}'.format(args.dirImgs))
        sys.exit(1)
    
    shutil.rmtree(args.dirResult, ignore_errors=True)
    
    start = datetime.datetime.now()
    allImgFPs = [ os.path.join(args.dirImgs, imgFN) for imgFN in os.listdir(args.dirImgs) ]
    ids    = [ i for i in range(len(allImgFPs)) ]
    logging.info('#image filepath: {}, e.g. {}'.format(len(allImgFPs), allImgFPs[:3]))
    logging.info('#image id: {}, e.g. {}'.format(len(ids), ids[:3]))
    logging.info('threshold distance of nearest neighbor: {}'.format(args.nnt))
    logging.info('--')

    part_imgFPs = []
    part_feavcts = []

    pool = Pool(processes=100)
    model = model_vgg16_fc2()
    bat_feavcts = []
    bat_imgFPs = []
    step = 1000
    for i in ids[::step]:
        logging.info('step - [{},{})'.format(i, i+step))
#        logging.info(allImgFPs[i:i+step])

        del tImg4DImgFN_list[:]
        pool.map(data_preprocess, allImgFPs[i:i+step])
       
        img4Ds, imgFPs = zip(*tImg4DImgFN_list)
        img4Ds = np.concatenate(img4Ds)
        img4Ds = vgg16.preprocess_input(img4Ds)
        feavcts = model.predict(img4Ds)
        bat_feavcts.extend( list(feavcts) )

        bat_imgFPs += imgFPs

        if 100 * 1000 <= len(bat_imgFPs):
            part_feavcts.append( np.asarray(bat_feavcts) )
            part_imgFPs.append( np.asarray(bat_imgFPs) )
            bat_feavcts = []
            bat_imgFPs = []
    
    if 0 < len(bat_feavcts):
        part_feavcts.append( np.asarray(bat_feavcts) )
        part_imgFPs.append( np.asarray(bat_imgFPs) )

    
    i = 0
    processed_set = set()
    for part_i in range(len(part_imgFPs)):
        logging.info('part - {}'.format(part_i))

        for imgFP, feavct in itertools.izip(part_imgFPs[part_i], part_feavcts[part_i]):
            i += 1
            if 0 == i % 1000:
                logging.info('image - {}'.format(i))

            if imgFP in processed_set:
                continue            
            
            nnFPs = nn_lte_threshold(feavct, imgFP, part_feavcts, part_imgFPs, args.nnt)
            nnFPs.append(imgFP)
            processed_set.update(nnFPs)

            rootDir = args.dirResult
            imgFN = os.path.split(imgFP)[-1]
            basename = os.path.splitext(imgFN)[0]
            imgTrainDir = os.path.join(rootDir, 'train', basename)

            for fp in nnFPs :
                cp_src2dst(fp, imgTrainDir)

    end = datetime.datetime.now()
    logging.info('total seconds: {}'.format((end-start).seconds))

