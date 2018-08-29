import os
import random
import logging
import numpy as np
import keras.applications.vgg16 as vgg16
import keras.layers as layers

from numpy import linalg as LA
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.datasets import cifar10



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

TRAIN_PATH = 'mini_train/train/'
IMG_H = 224
IMG_W = 224
DIM_OUTPUT = 4096
BATCH_SIZE = 64
AUG_SIZE_PER_IMAGE = 8
STEPS_PER_EPOCH = 500 #460
EPOCHS = 20



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square( K.maximum(margin - y_pred, 0) ))


def accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_network(input_shape):
    core_model = vgg16.VGG16(weights='imagenet', input_shape=input_shape)
    #-- set trainable layers
    for i, l in enumerate(core_model.layers):
        if i < 15:
            l.trainable = False
        else:
            l.trainable = True
        logging.info('{} {}.trainable: {}'.format(i, l.name, l.trainable))
    
    model = Model(inputs=core_model.input, outputs=core_model.get_layer('fc2').output)
    return model


def model_vgg16_fc2() :
    base_model = vgg16.VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    return model


def xy_generator(trainDir, img2vct_cnn, dist_identical_img=1.0e-3):
#    print 'xy_generator'

    imgFPs = [ os.path.join(trainDir, f) for f in os.listdir(trainDir) if os.path.isfile( os.path.join(trainDir, f)) ]

    imgDataGen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

#    xy_gen = imgDataGen.flow_from_directory(
#        'train_22850/train',
##        'mini_train/train/',
#        save_to_dir='tmp_mini_train_aug',
#        save_format='jpeg',
#        target_size=(IMG_H, IMG_W),
#        batch_size=BATCH_SIZE,
#        class_mode='binary'
#    )    
#    logging.debug(xy_gen.class_indices)


    #-- *2 due to the pairs of positive and negative samples
    size_anchor = BATCH_SIZE / (AUG_SIZE_PER_IMAGE*2)

    while True:
        x_pairs = []
        y_labels = []

        anchor_imgs = []
        anchor_ids = np.random.randint(len(imgFPs), size=size_anchor)
        imgfps = []
        for i in anchor_ids:
            imgfps.append(imgFPs[i])
            img = image.load_img(imgFPs[i], target_size=(224, 224))
            img_nda = image.img_to_array(img)
            anchor_imgs.append(img_nda)
        x = np.array(anchor_imgs)
        
        x = vgg16.preprocess_input(x)
        feavcts = img2vct_cnn.predict(x)

        labels = [None] * x.shape[0]
        norm_feavcts = LA.norm(feavcts, axis=1)
        for x_i, fv in enumerate(feavcts):
            sim = np.dot(feavcts, fv)
            sim = 1 - sim/LA.norm(fv)/norm_feavcts
###            sim_ids = np.argsort(sim)
            for j in range(x_i, len(sim)):
                if sim[j] <= dist_identical_img:
                    if labels[j] is None:
                        labels[j] = x_i
                        if j != x_i:
                            logging.info('{} {} are identical images'.format(imgfps[j], imgfps[x_i]))

        x_train = []
        y_train = []
        size_aug = AUG_SIZE_PER_IMAGE
        labels = set(labels)
        if len(labels) < 2:
            continue
        
        #-- duplicates for image augmentation 
        for x_i in labels:
            for i in range(size_aug):
                y_train.append(x_i)
                x_train.append(anchor_imgs[x_i])
        y_train = np.array(y_train)
        x_train = np.array(x_train)

        for x_aug_img, y_aug_labels in imgDataGen.flow(
                                        x_train, y_train,
                                        batch_size=BATCH_SIZE, 
#                                        save_to_dir='tmp_aug_sub22850',
#                                        save_prefix='jpeg'
                                        ):
            logging.debug('y_aug: {}'.format(y_aug_labels))

            labels = np.asarray(y_aug_labels)
            dedup_labels = np.asarray( list(set(y_aug_labels)) )
            logging.debug('num classes: {}'.format(len(dedup_labels)))
            if len(dedup_labels) < 2:
                logging.error('num classes: {} < 2'.foramt(len(dedup_labels)))
                break

            #-- indices of samples grouped by labels,
            #   e.g. [ array([2]), array([3]), array([6]), array([1, 5]), ... ]
            ids_labels = [ np.where(labels == l)[0] for l in dedup_labels ]
            #-- min size of sample indices arrays
            n = min( len(ids) for ids in ids_labels )
            logging.debug('min(len(indices of samples) for all labels): {}'.format(n))
            if n < 2:
                logging.warn('re-sampling (augmentation) again, due to {} < 2'.format(n))
                break

            num_classes = len(ids_labels)
            for i_l in range(num_classes):
                logging.debug('labels[{}]= {}, at indices {}'.format(i_l, dedup_labels[i_l], ids_labels[i_l]))
                for i in range(n):
                    #-- positive sample pair
                    j, k = ids_labels[i_l][i], ids_labels[i_l][(i+1) % n]
                    x_pairs += [ [x_aug_img[j], x_aug_img[k]] ]

                    r = random.randrange(1, num_classes)
                    i_l_neg = (i_l + r) % num_classes
                    j, k = ids_labels[i_l][i], ids_labels[i_l_neg][i]
                    #-- positive sample pair
                    x_pairs += [ [x_aug_img[j], x_aug_img[k]] ]
                    y_labels += [1, 0]

            break
                
        x_pairs_np = np.array(x_pairs)
        y_labels_np = np.array(y_labels)
        logging.debug(x_pairs_np.shape)
        logging.debug('x\'s shape:{}, y\'s shape:{}'.format(x_pairs_np.shape, y_labels_np.shape))

        logging.info(x_pairs_np.shape)
        yield [x_pairs_np[:,0], x_pairs_np[:,1]], y_labels_np
    


#import tensorflow as tf
#from keras import backend as K

if '__main__' ==  __name__:
#    gpu_options = tf.GPUOptions(allow_growth=True)
#    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#    K.set_session(sess)

    input_shape = (IMG_H, IMG_W, 3)
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    base_network = create_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    model.compile(
        loss = contrastive_loss,
#        loss='cosine_proximity',
#        optimizer=optimizers.RMSprop(lr=0.0005),
        optimizer = optimizers.Adam(lr=0.0005),
#        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics = [accuracy]
    )

    img2vct_cnn = model_vgg16_fc2()
    img2vct_cnn.predict( np.zeros((1, 224, 224, 3)) )
    model.fit_generator(
        xy_generator('img_sub2k', img2vct_cnn),
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = EPOCHS
    )

    modelJson = base_network.to_json()
    with open('img_sub2k.aug.simese.fc2-u{dim}.json'.format(dim=DIM_OUTPUT), "w") as m2j:
        m2j.write(modelJson)

    base_network.save_weights('img_sub2k.aug.simese.fc2-u{dim}.h5'.format(dim=DIM_OUTPUT))

