# -*- coding: utf-8 -*-

import os
import sys
import datetime
import argparse
import time

import numpy as np
from skimage import io
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import keras.applications.vgg16 as vgg16


SIZE_RS_LIST = 10

def data_preprocess(imgFP) :
    if imgFP.endswith('jpg'):
        img = image.load_img(imgFP, target_size=(224, 224))
        img3D = image.img_to_array(img)
        img4D = np.expand_dims(img3D, axis=0)
        imgFN = os.path.split(imgFP)[1]

        return (img3D, img4D, imgFN)


def load_fea_dataset(pathFeas) :
    imgFea1Ds = None
    imgBNs = None

    dirs = os.path.normpath(pathFeas).split(os.path.sep)
    if 0 < len(dirs):
        bnsFP = os.path.join(pathFeas, dirs[-1] + '.bns.npy')
        feasFP = os.path.join(pathFeas, dirs[-1] + '.feas.npy')
        if os.path.exists(feasFP) and os.path.exists(bnsFP):
            imgFea1Ds = np.load(feasFP)
            imgBNs = np.load(bnsFP)
        else:
            print '[ERROR] feature datasets are not found, {} {}'.format(bnsFP, feasFP)

    return imgFea1Ds, imgBNs 


##
## @pathFeas, the path of feature dataset 
## @imgFP, image file path, e.g. 'image2100000000/981887_L.jpg'
##
def search_sim_images(imgFP, imgFeaBN_trip, in_model=None) :
    start = datetime.datetime.now()
    img3D, img4D, imgFN = data_preprocess(imgFP)
    img4D = preprocess_input(img4D)
   
    #-- load model
    model = None
    if in_model:
        model = in_model
    else:
        model_base = VGG16(weights='imagenet')
        model = Model(inputs=model_base.input, outputs=model_base.get_layer('fc1').output)

    #-- image feature vector
    imgFea = model.predict(img4D)
    imgFea1D = imgFea.reshape(img4D.shape[0], 4096)
    print imgFea1D.shape
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)
    
    start = datetime.datetime.now()    
    imgFea1Ds, norm_imgFea1Ds, imgBNs = imgFeaBN_trip  ### load_fea_dataset(pathFeas)
    print imgFea1Ds.shape
    print imgBNs.shape
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)

    start = datetime.datetime.now()
    from scipy.spatial import distance    
#    knn = distance.cdist(imgFea1D, imgFea1Ds, 'cosine')
#    i_knn = np.argsort(knn[0])[0:SIZE_RS_LIST] 
#    simImgFNs = [ imgBNs[i] + '.jpg' for i in i_knn ]
    simImgFNs = cosine_similarity(imgFea1D, imgFea1Ds, norm_imgFea1Ds, imgBNs)
    simImgFNs = map(lambda f: f + '.jpg', simImgFNs)

    end = datetime.datetime.now()
    print 'cosine similarity: {} secs'.format((end-start).seconds)

    return simImgFNs

    
import logging
import logging.handlers
from scipy.spatial import distance
from numpy import linalg as LA
from multiprocessing import Pool, Manager

i2i_logger=logging.getLogger('I2ILogger')
i2i_logger.setLevel(logging.INFO)

ImgFea1Ds = None
Norm_ImgFea1Ds = None
ImgBNs = None
def cosine_similarity(imgFea1D, imgFea1Ds, norm_imgFea1Ds, imgBNs) :
    #-- produce the nearest neighbor once per feature vector to prevent out of memory during 
    #   dot product of large matries
###    v = ImgFea1Ds[i]
###    sim = np.dot(ImgFea1Ds, v)

    imgFea1D = imgFea1D.reshape( (imgFea1D.shape[-1],) )
    sim = np.dot(imgFea1Ds, imgFea1D)
    sim = 1 - sim/LA.norm(imgFea1D)/norm_imgFea1Ds
    i_ary = np.argsort(sim)[:20]

    top_l = map(lambda itop: imgBNs[itop], i_ary)
#    i2i_logger.info( '{}\t{}'.format(imgBNs[i], ','.join(top_l)) )
    
    return top_l


def similarity_matrix(pathFeas, outFP) :
    handler = logging.handlers.RotatingFileHandler(outFP)
    i2i_logger.addHandler(handler)

    global Norm_ImgFea1Ds
    global ImgFea1Ds
    global ImgBNs

    start = datetime.datetime.now()
    imgFea1Ds, imgBNs = load_fea_dataset(pathFeas)
    ImgFea1Ds = imgFea1Ds
    ImgBNs = imgBNs 
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)
    print imgFea1Ds.shape, imgFea1Ds.dtype
    
    start = datetime.datetime.now()
    Norm_ImgFea1Ds = LA.norm(imgFea1Ds, axis=1)
    pool = Pool(50)
    pool.map( cosine_similarity, range(imgFea1Ds.shape[0]) )
    end = datetime.datetime.now()
    

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='sp_name', help='sub-command help')

    parser_sm = subparsers.add_parser('sm', help='similarity matrix')
    parser_sm.add_argument("pathFeas", help="the directory of the image feature vectors")
    parser_sm.add_argument("outFP", help="the resuilt file path")

    parser_si = subparsers.add_parser('si', help='similar images')
    parser_si.add_argument("pathFeas", help="the directory of the image feature vectors")
    parser_si.add_argument('imgFP', help='the file path of an input image')

    args = parser.parse_args()

    if 'sm' == args.sp_name:
        similarity_matrix(args.pathFeas, args.outFP)
    elif 'si'  == args.sp_name:
        imgFea1Ds, imgBNs = load_fea_dataset(args.pathFeas)
        search_sim_images(args.imgFP, (imgFea1Ds, imgBNs))


