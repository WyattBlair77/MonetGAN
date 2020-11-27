import keras
from tensorflow.keras import layers

class Discriminator:
    def __init__(self, monet_ds, photo_ds, image_shape):
        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape

    # Our discriminator should have a funnel like shape to its architecture because we're taking many neurons and
    # slowly converting it into just a single neuron. So we'll down-sample the 256x256x3 image until it is a single
    # neuron, capable of telling us whether we're looking at a Monet or not
    def build(self):
        model = keras.models.Sequential()

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
        model.add(layers.InstanceNormalization())
        model.add(layers.ReLU())

        # Output
        model.add(layers.Conv2D(1, 4))

        return model

    def discriminator_loss(self, y_true, y_pred):
        loss = keras.backend.square(y_true-y_pred)  # (batch_size, 1)
        loss = keras.backend.sum(loss, axis=1)  # (batch_size,)
        return loss


