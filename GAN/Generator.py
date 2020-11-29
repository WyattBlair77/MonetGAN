import tensorflow as tf
import tensorflow_addons as tfa
import keras
from tensorflow.keras import layers


class Generator:
    def __init__(self, monet_ds, photo_ds, image_shape):

        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape

        self.OUTPUT_CHANNELS = 3

    # The generator should look like two funnels stuck together at their tips. Going to down-sample and then up-sample
    # to achieve this.
    def build(self):
        model = keras.models.Sequential()

        # Input layer
        model.add(layers.Input(shape=self.image_shape))

        # Down-sampling
        model.add(layers.Conv2D(64, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, 4))
        model.add(layers.LeakyReLU())

        # Up-sampling
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(256, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(128, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, 4))
        model.add(layers.LeakyReLU())

        # Output layer
        model.add(layers.Conv2DTranspose(self.OUTPUT_CHANNELS, activation='tanh'))

        return model

    def generate(self, photo):
        pass

    def generator_loss(self, y_true, y_pred):
        loss = keras.backend.square(y_true - y_pred)  # (batch_size, 1)
        loss = keras.backend.sum(loss, axis=1)  # (batch_size,)
        return loss
