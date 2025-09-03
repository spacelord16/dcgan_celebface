import tensorflow as tf
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
import config


def build_discriminator():
    """Builds the DCGAN Discriminator model."""
    model = tf.keras.Sequential(
        [
            Input(shape=(config.IM_SHAPE[0], config.IM_SHAPE[1], 3)),
            Conv2D(64, kernel_size=4, strides=2, padding="same"),
            LeakyReLU(0.2),
            Conv2D(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(256, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(1, kernel_size=4, strides=2, padding="same"),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    print("Discriminator model built.")
    model.summary()
    return model
