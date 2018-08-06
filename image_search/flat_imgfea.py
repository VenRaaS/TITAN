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
from keras.applications.resnet50 import ResNet50
import keras.applications.vgg16 as vgg16

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


def model_cust_vgg16(fn_mod) :
    base_model = vgg16.VGG16(include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    predictions = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(fn_mod)

    layer_output = model.get_layer('fc2').output

    model = Model(inputs=model.input, outputs=layer_output)
    dim_output = layer_output.shape[1] 
    return (model, dim_output)


def model_vgg16() :
    vgg_model = vgg16.VGG16(weights='imagenet')
    layer_output = vgg_model.get_layer('fc2').output

    model = Model(inputs=vgg_model.input, outputs=layer_output)
    dim_output = layer_output.shape[1] 
    return (model, dim_output)


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
    args = parser.parse_args()
    
    if not os.path.isdir(args.dirImgs):
        logging.error('an invalid dir: {}'.format(args.dirImgs))
        sys.exit(1)
    
    shutil.rmtree(args.dirFeaVcts, ignore_errors=True)
    
    start = datetime.datetime.now()
###    imgFPs = [ os.path.join(args.dirImgs, imgFN) for imgFN in os.listdir(args.dirImgs) ]
    imgFPs = lsfiles(args.dirImgs)
    ids    = [ i for i in range(len(imgFPs)) ]
    logging.info('image filepath - {}, e.g. {}'.format(len(imgFPs), imgFPs[:3]))
    logging.info('image id - {}, e.g. {}'.format(len(ids), ids[:3]))

    pool = Pool(processes=100)
    model, dim_output = model_vgg16()
#    model, dim_output = model_cust_vgg16('data_sub.block5-fc-u256.aug.h5')

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
        imgFea1Ds = imgFeas.reshape(img4Ds.shape[0], dim_output) # fc1, vgg16
#        imgFea1Ds = imgFeas.reshape(img4Ds.shape[0], 1*1*2048) # ResNet50

        for imgFN, imgFea in itertools.izip(imgFNs, imgFea1Ds):
            feaDir = args.dirFeaVcts
            if not os.path.exists(feaDir):
                os.makedirs(feaDir)

            feaFN = '{}.fea'.format(os.path.splitext(imgFN)[0])
            feaFP = os.path.join(feaDir, feaFN)
            np.save(feaFP, imgFea)

    #imgFNs = load_imgFNs('cc19049_559.vgg116.fn')
    #imgFPs = [ os.path.join('image1904900000', imgFN) for imgFN in imgFNs ]
    #img3Ds = load_img3Ds(imgFPs)
    end = datetime.datetime.now()
    logging.info('total seconds: {}'.format((end-start).seconds))

