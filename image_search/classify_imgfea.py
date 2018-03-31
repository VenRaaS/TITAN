# coding: utf-8

import os
import sys
import datetime
import shutil
from multiprocessing import Pool, Manager

from keras.preprocessing import image
import numpy as np
from skimage import io
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50


FEA_VCT_DIR = 'FeaVCT_cc21000'

tImg4DImgFN_list = Manager().list()
def data_preprocess(imgFP):
    if imgFP.endswith('jpg'):
        img = image.load_img(imgFP, target_size=(224, 224))
        img3D = image.img_to_array(img)
        img4D = np.expand_dims(img3D, axis=0)
        imgFN = os.path.split(imgFP)[1]
        tImg4DImgFN_list.append( (img4D, imgFN) )

if '__main__' == __name__:
    shutil.rmtree(FEA_VCT_DIR, ignore_errors=True)
    
    start = datetime.datetime.now()
    imgFPs = [ os.path.join('image2100000000', imgFN) for imgFN in os.listdir("image2100000000") ]
#    imgFPs = [ os.path.join('image1904900000', imgFN) for imgFN in os.listdir('image1904900000') ]
    ids    = [ i for i in range(len(imgFPs)) ]

    pool = Pool(processes=100)

    modelCls = ResNet50(weights='imagenet', include_top=True)
    modelFea = ResNet50(weights='imagenet', include_top=False)

    step = 1000 
    if len(imgFPs) < step:
        step = len(imgFPs)/2
    for i in ids[::step]:
        print 'step - [{},{})'.format(i, i+step)

        del tImg4DImgFN_list[:]
        pool.map(data_preprocess, imgFPs[i:i+step])
        
        img4Ds, imgFNs = zip(*tImg4DImgFN_list)
        img4Ds = np.concatenate(img4Ds)
        img4Ds = preprocess_input(img4Ds)

        #-- top predicted classes
        topClsPreds = modelCls.predict(img4Ds)
        topClsPreds = decode_predictions(topClsPreds, top=3)

        #-- image feature vector
        imgFeas = modelFea.predict(img4Ds)
        imgFea1Ds = imgFeas.reshape(img4Ds.shape[0], 1*1*2048) #ResNet50

        #-- save feature vector to top5 predictions (classes)
        tImgFNtTopCImgFea_list = zip(imgFNs, topClsPreds, imgFea1Ds)    
        for imgFN, topCs, imgFea in tImgFNtTopCImgFea_list:
            for c in topCs:
                feaDir = os.path.join(FEA_VCT_DIR, c[0])
                if not os.path.exists(feaDir):
                    os.makedirs(feaDir)

                feaFN = '{}.fea'.format(os.path.splitext(imgFN)[0])
                feaFP = os.path.join(feaDir, feaFN)        
                np.save(feaFP, imgFea)

    #imgFNs = load_imgFNs('cc19049_559.vgg116.fn')
    #imgFPs = [ os.path.join('image1904900000', imgFN) for imgFN in imgFNs ]
    #img3Ds = load_img3Ds(imgFPs)
    end = datetime.datetime.now()
    print((end-start).seconds)

