import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Generator:
    def __init__(self, monet_ds, photo_ds, image_shape, batch_size):

        self.model = None

        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.OUTPUT_CHANNELS = 3

        self.input_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]

    # The generator should look like two funnels stuck together at their tips. Going to down-sample and then up-sample
    # to achieve this.
    def build(self):
        model = tf.keras.Sequential(name='Generator')

        # Input layer
        model.add(layers.Conv2D(32, (3, 3), input_shape=self.image_shape))

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
        model.add(layers.Dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.Dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.Dropout(0.5))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(512, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(256, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(128, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(32, 4))
        model.add(layers.LeakyReLU())
        model.add(layers.ZeroPadding2D((1, 1)))

        # Output layer
        model.add(layers.Conv2DTranspose(self.OUTPUT_CHANNELS, kernel_size=1, activation='tanh'))

        model.build(input_shape=self.image_shape)
        model.summary()

        self.model = model

    def generate(self, photo):

        if self.model is None:
            raise ValueError('The model has not been built yet. Use Generator.build() '
                             'before using Generator.generate().')

        model_evaluation = self.model(photo, training=False)
        model_evaluation = model_evaluation.numpy()
        model_evaluation = model_evaluation.astype('unit8')
        model_evaluation.reshape(self.image_shape)
        model_evaluation = np.squeeze(model_evaluation, axis=0)

        return model_evaluation

