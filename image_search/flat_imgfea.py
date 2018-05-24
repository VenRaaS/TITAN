# coding: utf-8

import argparse
import os
import sys
import datetime
import shutil
import itertools
from multiprocessing import Pool, Manager

from keras.preprocessing import image
import numpy as np
from skimage import io
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import keras.applications.vgg16 as vgg16


tImg4DImgFN_list = Manager().list()
def data_preprocess(imgFP):
    try:
        if imgFP.endswith('jpg'):
            img = image.load_img(imgFP, target_size=(224, 224))
            img3D = image.img_to_array(img)
            img4D = np.expand_dims(img3D, axis=0)
            imgFN = os.path.split(imgFP)[1]
            tImg4DImgFN_list.append( (img4D, imgFN) )
    except Exception as e:
        print str(e)

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirImgs", help="the directory of the source images")
    parser.add_argument("dirFeaVcts", help="the directory for the output feature vectors of the given images")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dirImgs):
        print 'an invalid dir: {}'.format(args.dirImgs)
        sys.exit(1)
    
    shutil.rmtree(args.dirFeaVcts, ignore_errors=True)
    
    start = datetime.datetime.now()
    imgFPs = [ os.path.join(args.dirImgs, imgFN) for imgFN in os.listdir(args.dirImgs) ]
    ids    = [ i for i in range(len(imgFPs)) ]
    print 'image filepath - {}, e.g. {}'.format(len(imgFPs), imgFPs[:3])
    print 'image id - {}, e.g. {}'.format(len(ids), ids[:3])

    pool = Pool(processes=100)
    vgg_model = vgg16.VGG16(weights='imagenet')
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc1').output)

    step = 1000 
    if len(imgFPs) < step:
        step = len(imgFPs)/2
    for i in ids[::step] :
        print 'step - [{},{})'.format(i, i+step)

        del tImg4DImgFN_list[:]
        pool.map(data_preprocess, imgFPs[i:i+step])
        
        img4Ds, imgFNs = zip(*tImg4DImgFN_list)
        img4Ds = np.concatenate(img4Ds)
        img4Ds = vgg16.preprocess_input(img4Ds)

        #-- image feature vector
        imgFeas = model.predict(img4Ds)
        imgFea1Ds = imgFeas.reshape(img4Ds.shape[0], 4096) # fc1, vgg16
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
    print((end-start).seconds)

