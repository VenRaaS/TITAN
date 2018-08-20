import random
import logging
import numpy as np
import keras.applications.vgg16 as vgg16
import keras.layers as layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras import backend as K



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

TRAIN_PATH = 'mini_train/train/'
IMG_H = 224
IMG_W = 224
DIM_OUTPUT = 4096
BATCH_SIZE = 32
STEPS_PER_EPOCH = 50 #460


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
        if i < 20:
            l.trainable = False
        else:
            l.trainable = True
        logging.info('{} {}.trainable: {}'.format(i, l.name, l.trainable))
    
    model = Model(inputs=core_model.input, outputs=core_model.get_layer('fc2').output)
    return model


def xy_generator():
#    print 'xy_generator'

    imgDataGen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    xy_gen = imgDataGen.flow_from_directory(
        'mini_train/train/',
#        save_to_dir='tmp_mini_train_aug',
#        save_format='jpeg',
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    logging.info(xy_gen.class_indices)

    for x_gen_img, y_gen_labels in xy_gen:
        x_pairs = []
        y_labels = []

        labels = np.asarray(y_gen_labels)
        dedup_labels = np.asarray( list(set(y_gen_labels)) )
        logging.info('num classes: {}'.format(len(dedup_labels)))
        if len(dedup_labels) < 2:
            logging.error('num classes {} < 2'.foramt(len(dedup_labels)))

        #-- indices of samples grouped by labels,
        #   e.g. [ array([2]), array([3]), array([6]), array([1, 5]), ... ]
        ids_labels = [ np.where(labels == l)[0] for l in dedup_labels ]
        #-- min size of sample indices arrays
        n = min( len(ids) for ids in ids_labels )
        ub = n - 1
        logging.info('min(len(sample indices) for all labels): {}, and upper bound of sample indices: {}'.format(n, ub))
        if n < 2:
            logging.info('resampling again, due to {} < 2'.format(n))
            continue

#        print y_gen_labels
        num_classes = len(ids_labels)
        for i_l in range(num_classes):
#            logging.info('labels[{}]= {}, at indices {}'.format(i_l, dedup_labels[i_l], ids_labels[i_l]))
            
            for i in range(ub):
                #-- positive sample pair
                j, k = ids_labels[i_l][i], ids_labels[i_l][i + 1]
                x_pairs += [ [x_gen_img[j], x_gen_img[k]] ]

                r = random.randrange(1, num_classes)
                i_l_neg = (i + r) % num_classes
                j, k = ids_labels[i_l][i], ids_labels[i_l_neg][i]
                #-- positive sample pair
                x_pairs += [ [x_gen_img[j], x_gen_img[k]] ]
                y_labels += [1, 0]

        x_pairs_np = np.array(x_pairs)
        y_labels_np = np.array(y_labels)
        logging.info('x\'s shape:{}, y\'s shape:{}'.format(x_pairs_np.shape, y_labels_np.shape))
        yield [x_pairs_np[:,0], x_pairs_np[:,1]], y_labels_np
    

if '__main__' ==  __name__:
###    gen = xy_generator()
###    for x,y in gen:
###        print x, y
###    exit()

    input_shape = (IMG_H, IMG_W, 3)
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    base_network = create_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    model.compile(
        loss=contrastive_loss,
#        loss='cosine_proximity',
        optimizer=optimizers.RMSprop(),
#        optimizer='adam',
#        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=[accuracy]
    )

    model.fit_generator(
        xy_generator(),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=10
    )

    modelJson = base_network.to_json()
    with open('mini_train.aug.simese.fc2-u{dim}.json'.format(dim=DIM_OUTPUT), "w") as m2j:
        m2j.write(modelJson)

    base_network.save_weights('mini_train.aug.simese.fc2-u{dim}.h5'.format(dim=DIM_OUTPUT))

