# coding: utf-8

import os
import sys
import datetime
from multiprocessing import Pool, Manager
import itertools

import numpy as np
from skimage import io
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


FEA_VCT_DIR = 'FeaVCT_cc21000_flat'
SIZE_RS_LIST = 10

def data_preprocess(imgFP) :
    if imgFP.endswith('jpg'):
        img = image.load_img(imgFP, target_size=(224, 224))
        img3D = image.img_to_array(img)
        img4D = np.expand_dims(img3D, axis=0)
        imgFN = os.path.split(imgFP)[1]

        return (img3D, img4D, imgFN)

##
## @imgFP, image file path, e.g. 'image2100000000/981887_L.jpg'
##
def search_sim_images(imgFP) :
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
    imgFea1Ds = None
    imgBNs = None
    
    bnsFP = os.path.join(FEA_VCT_DIR, FEA_VCT_DIR + '.bns.npy')
    feasFP = os.path.join(FEA_VCT_DIR, FEA_VCT_DIR + '.feas.npy')        
    if os.path.exists(feasFP) and os.path.exists(bnsFP):
        imgFea1Ds = np.load(feasFP)
        imgBNs = np.load(bnsFP)
            
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
