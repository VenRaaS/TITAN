# coding: utf-8

import argparse
import os
import sys
import datetime
import shutil
import itertools
import logging
from fnmatch import fnmatch
from multiprocessing import Pool, Manager

from keras.preprocessing import image
import keras.layers as layers
import numpy as np
from skimage import io
from keras.models import Model
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
import keras.applications.vgg16 as vgg16

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


def load_model_arch_weight(arch_fn, weight_fn) :
    model = None
    with open(arch_fn, 'r') as mj_f:
        json_str = mj_f.read()
        model = model_from_json(json_str)
    
    model.load_weights(weight_fn)
    
    return model


def model_vgg16_fc2() :
    vgg_model = vgg16.VGG16(weights='imagenet')
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

    return model


def lsfiles(d, pattern='*.jpg') :
    fn_list = os.listdir(d)

    fp_list = []
    for fn in fn_list:
        path = os.path.join(d, fn)
        if os.path.isdir(path):
            fp_list += lsfiles(path)
        else:
            if fnmatch(path, pattern):
                fp_list.append(path)

    return fp_list


tImg4DImgFN_list = Manager().list()
def data_preprocess(imgFP) :
    try:
        if imgFP.endswith('jpg'):
            img = image.load_img(imgFP, target_size=(224, 224))
            img3D = image.img_to_array(img)
            img4D = np.expand_dims(img3D, axis=0)
            imgFN = os.path.split(imgFP)[1]
            tImg4DImgFN_list.append( (img4D, imgFN) )
    except Exception as e:
        logging.error(str(e))


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirImgs", help="the directory of the source images")
    parser.add_argument("dirFeaVcts", help="the directory for the output feature vectors of the given images")
    parser.add_argument("--m", help="the file base (stem) name of the architecture and weight model file")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dirImgs):
        logging.error('an invalid dir: {}'.format(args.dirImgs))
        sys.exit(1)
    
    shutil.rmtree(args.dirFeaVcts, ignore_errors=True)
    
    start = datetime.datetime.now()
    imgFPs = lsfiles(args.dirImgs)
    ids    = [ i for i in range(len(imgFPs)) ]
    logging.info('image filepath - {}, e.g. {}'.format(len(imgFPs), imgFPs[:3]))
    logging.info('image id - {}, e.g. {}'.format(len(ids), ids[:3]))
    logging.info('image id - {}, e.g. {}'.format(len(ids), ids[:3]))
   
    model = None 
    if args.m:
        fn_model_arch = args.m + '.json'
        fn_model_weight = args.m + '.h5'
        logging.info('model files: {}, {}'.format(fn_model_arch, fn_model_weight))
        model = load_model_arch_weight(fn_model_arch, fn_model_weight)
    else:
        model = model_vgg16_fc2()

    model.summary()
    dim_output = model.outputs[0].shape[1]
    logging.info('dim of output is {}'.format(dim_output))

    pool = Pool(processes=100)
    step = 1000 
    if len(imgFPs) < step:
        step = len(imgFPs)/2
    for i in ids[::step] :
        logging.info('step - [{},{})'.format(i, i+step))

        del tImg4DImgFN_list[:]
        pool.map(data_preprocess, imgFPs[i:i+step])
        
        img4Ds, imgFNs = zip(*tImg4DImgFN_list)
        img4Ds = np.concatenate(img4Ds)
        img4Ds = vgg16.preprocess_input(img4Ds)

        #-- image feature vector
        imgFeas = model.predict(img4Ds)
        imgFea1Ds = imgFeas.reshape(img4Ds.shape[0], dim_output)

        for imgFN, imgFea in itertools.izip(imgFNs, imgFea1Ds):
            feaDir = args.dirFeaVcts
            if not os.path.exists(feaDir):
                os.makedirs(feaDir)

            feaFN = '{}.fea'.format(os.path.splitext(imgFN)[0])
            feaFP = os.path.join(feaDir, feaFN)
            np.save(feaFP, imgFea)

    end = datetime.datetime.now()
    logging.info('total seconds: {}'.format((end-start).seconds))

