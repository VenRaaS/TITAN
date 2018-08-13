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


if '__main__' == __name__ :
    parser = argparse.ArgumentParser()
    parser.add_argument("dirImgs", help="the directory of the source images")
    parser.add_argument("dirResult", help="the directory for the categorized results")
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

    pool = Pool(processes=100)
    model = vgg16.VGG16(weights='imagenet')

    step = 1000 
    if len(allImgFPs) < step:
        step = len(allImgFPs)/2
    for i in ids[::step] :
        logging.info('step - [{},{})'.format(i, i+step))
#        logging.info(allImgFPs[i:i+step])

        del tImg4DImgFN_list[:]
        pool.map(data_preprocess, allImgFPs[i:i+step])
       
        img4Ds, imgFPs = zip(*tImg4DImgFN_list)
        img4Ds = np.concatenate(img4Ds)
        img4Ds = vgg16.preprocess_input(img4Ds)

        preds = model.predict(img4Ds)
        #-- decode the results into a list of tuples [(class, description, probability), ...]
        triCDPs = decode_predictions(preds, top=1)
#        logging.info('{}'.format(triCDPs))

        for imgFP, triCDP in itertools.izip(imgFPs, triCDPs):
            feaDir = args.dirResult
            imgTrainDir = os.path.join(feaDir, 'train', '-'.join(triCDP[0][0:2]))
            cp_src2dst(imgFP, imgTrainDir)


    end = datetime.datetime.now()
    logging.info('total seconds: {}'.format((end-start).seconds))

