import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Reshape, Conv2DTranspose, 
                                     BatchNormalization, LeakyReLU)
import config

def build_generator():
    """Builds the DCGAN Generator model."""
    model = tf.keras.Sequential([
        Input(shape=(config.LATENT_DIM,)),
        Dense(4*4*config.LATENT_DIM),
        Reshape((4, 4, config.LATENT_DIM)),

        Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(3, kernel_size=4, strides=2, activation='tanh', padding='same'),
    ], name='generator')
    
    print("✅ Generator model built.")
    model.summary()
    return model