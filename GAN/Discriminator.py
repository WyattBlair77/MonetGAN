import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential, Model


class Discriminator:
    def __init__(self, monet_ds, photo_ds, image_shape):
        self.model = None

        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape

    # Our discriminator should have a funnel like shape to its architecture because we're taking many neurons and
    # slowly converting it into just a single neuron. So we'll down-sample the 256x256x3 image until it is a single
    # neuron, capable of telling us whether we're looking at a Monet or not
    def build(self):
        model = tf.keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=self.image_shape, name='photo_input'))

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
        model.add(layers.Conv2D(1, 4))

        model.summary()
        model.build(input_shape=self.image_shape)

        self.model = model
