from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
import keras.applications.vgg16 as vgg16
import keras.layers as layers
import numpy as np


DIM_OUTPUT = 1024

BATCH_SIZE = 64

IMG_H=224
IMG_W=224


if '__main__' ==  __name__:
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, 3))

    x = base_model.output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(DIM_OUTPUT, activation='relu', name='fc1')(x)
    x = layers.Dense(DIM_OUTPUT, activation='relu', name='fc2')(x)
    predictions = layers.Dense(10, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    #-- set trainable layers
    for l in base_model.layers:
        l.trainable = False
    
    for i, l in  enumerate(model.layers):
        print '{} {}.trainable: {}'.format(i, l.name, l.trainable)
#    model.load_weights('data_sub.fc-u256.aug.out')

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
#        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
 
    imgdatagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    train_generator = imgdatagen.flow_from_directory(
        'data_sub/train',
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=460,
        epochs=100
    )

    for i, l in  enumerate(model.layers):
        if i < 15:
            l.trainable = False
        else:
            l.trainable = True
        print '{} {}.trainable: {}'.format(i, l.name, l.trainable)

    model.compile(
        loss='categorical_crossentropy',
#        optimizer='adam',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=460,
        epochs=100
    )

#    model.save_weights('data_sub.fc-u{dim}.aug.out'.format(dim=DIM_OUTPUT))
    model.save_weights('data_sub.block5.fc-u{dim}.aug.h5'.format(dim=DIM_OUTPUT))


