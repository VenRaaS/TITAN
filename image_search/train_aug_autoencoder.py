from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
import keras.applications.vgg16 as vgg16
import keras.layers as layers
import numpy as np


IMG_H = 224
IMG_W = 224
DIM_OUTPUT = 4096
BATCH_SIZE = 64


dec_base_model = vgg16.VGG16(weights='imagenet')
dec_model = Model(inputs=dec_base_model.input, outputs=dec_base_model.get_layer('fc2').output)
#-- https://github.com/keras-team/keras/issues/2397#issuecomment-354061212
init_pred2avoiderr = dec_model.predict( np.zeros((1, 224, 224, 3)) )

def xy_generator():
    global dec_model

    imgDataGen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    xy_gen = imgDataGen.flow_from_directory(
        'data_sub/train',
#        'data_mini/train',
#        save_to_dir='tmp_train_aug',
#        save_format='jpeg',
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        class_mode='input'
    )

    for x_bat, y_bat in xy_gen:
        y_bat_feavct = dec_model.predict(y_bat)
        yield (x_bat, y_bat_feavct)
    

if '__main__' ==  __name__:
    encoder_base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, 3))
    x = encoder_base_model.output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(DIM_OUTPUT, activation='relu', name='fc1')(x)
    predictions = layers.Dense(DIM_OUTPUT, activation='relu', name='fc2')(x)
    encoder_model = Model(inputs=encoder_base_model.input , outputs=predictions)

    #-- set trainable layers
    for i, l in enumerate(encoder_model.layers):
        if i < 15:
            l.trainable = False
        else:
            l.trainable = True
        print '{} {}.trainable: {}'.format(i, l.name, l.trainable)

    encoder_model.compile(
       loss='mse',
#        loss='cosine_proximity',
        optimizer='adam',
#        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    )

    encoder_model.fit_generator(
        xy_generator(),
        steps_per_epoch=460,
        epochs=50
    )

    modelJson = encoder_model.to_json()
    with open('data_sub.aug.autoencoder.block5.fc2-u{dim}.json'.format(dim=DIM_OUTPUT), "w") as m2j:
        m2j.write(modelJson)

#   encoder_model.save_weights('data_sub.fc-u{dim}.aug.out'.format(dim=DIM_OUTPUT))
    encoder_model.save_weights('data_sub.aug.autoencoder.block5.fc2-u{dim}.h5'.format(dim=DIM_OUTPUT))

