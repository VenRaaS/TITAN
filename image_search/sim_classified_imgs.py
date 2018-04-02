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


FEA_VCT_DIR = 'FeaVCT_cc21000'


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

    modelCls = ResNet50(weights='imagenet', include_top=True)
    modelFea = ResNet50(weights='imagenet', include_top=False)

    #-- image feature vector
    imgFea = modelFea.predict(img4D)
    imgFea1D = imgFea.reshape(img4D.shape[0], 1*1*2048) #ResNet50

    #-- top predicted classes
    topClsPreds = modelCls.predict(img4D)
    topClsPreds = decode_predictions(topClsPreds, top=3)
    print topClsPreds

    end = datetime.datetime.now()
    print((end-start).seconds)


    imgBN2Fea1D_dic = {}
    start = datetime.datetime.now()
    for tTopC in topClsPreds[0]:
        clsID = tTopC[0] 
        feaDir = os.path.join(FEA_VCT_DIR, clsID)    
        bnsFP = os.path.join(feaDir, clsID + '.bns.npy')
        feasFP = os.path.join(feaDir, clsID + '.feas.npy')
        
        fea_na = np.load(feasFP)
        bn_na = np.load(bnsFP)
        
        for basename, fea in itertools.izip(bn_na, fea_na):
            imgBN2Fea1D_dic[basename] = fea    

    end = datetime.datetime.now()
    print '{} secs'.format((end-start).seconds)
    print len(imgBN2Fea1D_dic)


    from scipy.spatial import distance

    imgFea1Ds = imgBN2Fea1D_dic.values()
    knn = distance.cdist(imgFea1D, imgFea1Ds, 'cosine')
    i_knn = np.argsort(knn[0])[0:10]
    imgBNs = imgBN2Fea1D_dic.keys()

    simImgFNs = [ imgBNs[i] + '.jpg' for i in i_knn ]

    return simImgFNs
    
