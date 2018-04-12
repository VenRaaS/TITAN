# coding: utf-8

import os
import sys
import datetime
import argparse
import time

import numpy as np
from skimage import io
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


SIZE_RS_LIST = 10

def data_preprocess(imgFP) :
    if imgFP.endswith('jpg'):
        img = image.load_img(imgFP, target_size=(224, 224))
        img3D = image.img_to_array(img)
        img4D = np.expand_dims(img3D, axis=0)
        imgFN = os.path.split(imgFP)[1]

        return (img3D, img4D, imgFN)


def load_model(dirModel) :
    imgFea1Ds = None
    imgBNs = None

    dirs = os.path.normpath(dirModel).split(os.path.sep)
    if 0 < len(dirs):
        bnsFP = os.path.join(dirModel, dirs[0] + '.bns.npy')
        feasFP = os.path.join(dirModel, dirs[0] + '.feas.npy')
        if os.path.exists(feasFP) and os.path.exists(bnsFP):
            imgFea1Ds = np.load(feasFP)
            imgBNs = np.load(bnsFP)

    return imgFea1Ds, imgBNs 

##
## @imgFP, image file path, e.g. 'image2100000000/981887_L.jpg'
##
def search_sim_images(dirModel, imgFP) :
    start = datetime.datetime.now()
    img3D, img4D, imgFN = data_preprocess(imgFP)
    img4D = preprocess_input(img4D)
    
    #-- image feature vector
    modelFea = ResNet50(weights='imagenet', include_top=False)
    imgFea = modelFea.predict(img4D)
    imgFea1D = imgFea.reshape(img4D.shape[0], 1*1*2048) #ResNet50
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)
    
    start = datetime.datetime.now()    
    imgFea1Ds, imgBNs = load_model(dirModel)
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)

    start = datetime.datetime.now()
    from scipy.spatial import distance    
    knn = distance.cdist(imgFea1D, imgFea1Ds, 'cosine')
    i_knn = np.argsort(knn[0])[0:SIZE_RS_LIST] 

    simImgFNs = [ imgBNs[i] + '.jpg' for i in i_knn ]
    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)    

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
def cosine_similarity(i) :
    #-- produce the nearest neighbor once per feature vector to prevent out of memory during 
    #   dot product of large matries
    v = ImgFea1Ds[i]
    sim = np.dot(ImgFea1Ds, v)
    sim = 1 - sim/LA.norm(v)/Norm_ImgFea1Ds
    i_ary = np.argsort(sim)[:20]

    top_l = map(lambda itop: ImgBNs[itop], i_ary)
    i2i_logger.info( '{}\t{}'.format(ImgBNs[i], ','.join(top_l)) )


def similarity_matrix(dirModel, outFP) :
    handler = logging.handlers.RotatingFileHandler(outFP)
    i2i_logger.addHandler(handler)

    global Norm_ImgFea1Ds
    global ImgFea1Ds
    global ImgBNs

    start = datetime.datetime.now()
    imgFea1Ds, imgBNs = load_model(dirModel)
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
    parser_sm.add_argument("dirModel", help="the directory of the image feature vectors")
    parser_sm.add_argument("outFP", help="the resuilt file path")

    parser_si = subparsers.add_parser('si', help='similar images')
    parser_si.add_argument("dirModel", help="the directory of the image feature vectors")
    parser_si.add_argument('imgFP', help='the file path of an input image')

    args = parser.parse_args()

    if 'sm' == args.sp_name:
        similarity_matrix(args.dirModel, args.outFP)
    elif 'si'  == args.sp_name:
        search_sim_images(args.dirModel, args.imgFP)


