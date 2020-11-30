import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential, Model


class Discriminator:
    def __init__(self, monet_ds, photo_ds, image_shape, batch_size):
        self.model = None

        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape
        self.batch_size = batch_size

        self.input_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]


    # Our discriminator should have a funnel like shape to its architecture because we're taking many neurons and
    # slowly converting it into just a single neuron. So we'll down-sample the 256x256x3 image until it is a single
    # neuron, capable of telling us whether we're looking at a Monet or not
    def build(self):
        model = tf.keras.Sequential(name='Discriminator')

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

        # Normalize everything
        model.add(layers.LayerNormalization())
        model.add(layers.ReLU())

        # Output
        model.add(layers.Dense(1))

        model.summary()
        model.build(input_shape=self.image_shape)

        self.model = model
