import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage import transform, measure

tf.keras.backend.set_floatx('float16')

class UNet(keras.Model):
    def __init__(self, weights='weights/unet_yeast_bf_float16.h5'):
        input_size=(None, None, 1)
        d0a = layers.Input(input_size, name='input')
        d0b = layers.Conv2D(64, 3, padding='same', name='conv_d0a-b')(d0a)
        d0b = layers.ReLU(negative_slope=0.1, name='relu_d0b')(d0b)
        d0c = layers.Conv2D(64, 3, padding='same', name='conv_d0b-c')(d0b)
        d0c = layers.ReLU(negative_slope=0.1, name='relu_d0c')(d0c)

        d1a = layers.MaxPooling2D(pool_size=2, name='pool_d0c-1a')(d0c)
        d1b = layers.Conv2D(128, 3, padding='same', name='conv_d1a-b')(d1a)
        d1b = layers.ReLU(negative_slope=0.1, name='relu_d1b')(d1b)
        d1c = layers.Conv2D(128, 3, padding='same', name='conv_d1b-c')(d1b)
        d1c = layers.ReLU(negative_slope=0.1, name='relu_d1c')(d1c)

        d2a = layers.MaxPooling2D(pool_size=2, name='pool_d1c-2a')(d1c)
        d2b = layers.Conv2D(256, 3, padding='same', name='conv_d2a-b')(d2a)
        d2b = layers.ReLU(negative_slope=0.1, name='relu_d2b')(d2b)
        d2c = layers.Conv2D(256, 3, padding='same', name='conv_d2b-c')(d2b)
        d2c = layers.ReLU(negative_slope=0.1, name='relu_d2c')(d2c)

        d3a = layers.MaxPooling2D(pool_size=2, name='pool_d2c-3a')(d2c)
        d3b = layers.Conv2D(512, 3, padding='same', name='conv_d3a-b')(d3a)
        d3b = layers.ReLU(negative_slope=0.1, name='relu_d3b')(d3b)
        d3c = layers.Conv2D(512, 3, padding='same', name='conv_d3b-c')(d3b)
        d3c = layers.ReLU(negative_slope=0.1, name='relu_d3c')(d3c)
        d3c = layers.Dropout(0.5, name='dropout_d3c')(d3c)

        d4a = layers.MaxPooling2D(pool_size=2, name='pool_d3c-4a')(d3c)
        d4b = layers.Conv2D(1024, 3, padding='same', name='conv_d4a-b')(d4a)
        d4b = layers.ReLU(negative_slope=0.1, name='relu_d4b')(d4b)
        d4c = layers.Conv2D(1024, 3, padding='same', name='conv_d4b-c')(d4b)
        d4c = layers.ReLU(negative_slope=0.1, name='relu_d4c')(d4c)
        d4c = layers.Dropout(0.5, name='dropout_d4c')(d4c)

        u3a = layers.Conv2DTranspose(512, 2, strides=2, name='upconv_d4c_u3a')(d4c)
        u3a = layers.ReLU(negative_slope=0.1)(u3a)
        u3b = layers.Concatenate(name='concat_d3c_u3a-b')([u3a, d3c])
        u3c = layers.Conv2D(512, 3, padding='same', name='conv_u3b-c')(u3b)
        u3c = layers.ReLU(negative_slope=0.1, name='relu_u3c')(u3c)
        u3d = layers.Conv2D(512, 3, padding='same', name='conv_u3c-d')(u3c)
        u3d = layers.ReLU(negative_slope=0.1, name='relu_u3d')(u3d)

        u2a = layers.Conv2DTranspose(256, 2, strides=2, padding='same', name='upconv_u3d_u2a')(u3d)
        u2a = layers.ReLU(negative_slope=0.1)(u2a)
        u2b = layers.Concatenate(name='concat_d2c_u2a-b')([u2a, d2c])
        u2c = layers.Conv2D(256, 3, padding='same', name='conv_u2b-c')(u2b)
        u2c = layers.ReLU(negative_slope=0.1, name='relu_u2c')(u2c)
        u2d = layers.Conv2D(256, 3, padding='same', name='conv_u2c-d')(u2c)
        u2d = layers.ReLU(negative_slope=0.1, name='relu_u2d')(u2d)

        u1a = layers.Conv2DTranspose(128, 2, strides=2, padding='same',name='upconv_u2d_u1a')(u2d)
        u1a = layers.ReLU(negative_slope=0.1)(u1a)
        u1b = layers.Concatenate(name='concat_d1c_u1a-b')([u1a, d1c])
        u1c = layers.Conv2D(128, 3, padding='same', name='conv_u1b-c')(u1b)
        u1c = layers.ReLU(negative_slope=0.1, name='relu_u1c')(u1c)
        u1d = layers.Conv2D(128, 3, padding='same', name='conv_u1c-d')(u1c)
        u1d = layers.ReLU(negative_slope=0.1, name='relu_u1d')(u1d)

        u0a = layers.Conv2DTranspose(128, 2, strides=2, padding='same', name='upconv_u1d_u0a')(u1d)
        u0a = layers.ReLU(negative_slope=0.1, name='relu_u0a')(u0a)
        u0b = layers.Concatenate(name='concat_d0c_u0a-b')([u0a, d0c])
        u0c = layers.Conv2D(128, 3, padding='same', name='conv_u0b-c')(u0b)
        u0c = layers.ReLU(negative_slope=0.1, name='relu_u0c')(u0c)
        u0d = layers.Conv2D(128, 3, padding='same', name='conv_u0c-d')(u0c)
        u0d = layers.ReLU(negative_slope=0.1, name='relu_u0d')(u0d)

        score = layers.Conv2D(2, 1, name='conv_u0d-score')(u0d)
        score = layers.Softmax(name='score')(score)

        super().__init__(inputs=d0a, outputs=score, name='unet_yeast_bf')

        self.build(input_shape=input_size)

        if weights:
            current_dir = Path(__file__).resolve().parent
            weights_path = current_dir / weights
            if not os.path.exists(weights_path):
                raise ValueError('weights file not found')
            
            self.load_weights(weights_path)
        
    def label(self, score, score_threshold=0.95):
        mask = score>0.5
        labels = measure.label(mask)

        df = pd.DataFrame(measure.regionprops_table(labels, score, properties=('label', 'intensity_mean')))
        for i in list(df[df.intensity_mean <= 0.95].label):
            labels[labels==i] = 0
        
        mask = labels > 0
        labels = measure.label(mask)
        return labels

    def segment_and_label(self, im, is_binned=False):
        if len(im.shape) != 2:
            raise ValueError('expecting 2D images')
        
        im_min = im.min()
        im_max = im.max()
        imnorm = np.array((im - im_min) / (im_max - im_min), dtype=np.float16)
        
        if is_binned:
            imnorm = transform.rescale(imnorm, 2)

        score = self.predict(imnorm[np.newaxis, :, :, np.newaxis], verbose=False)
        score = score[0,:,:,1]

        if is_binned:
            score = transform.rescale(score, 0.5)
            
        labels = self.label(score)
        return score, labels
