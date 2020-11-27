import tensorflow as tf
print("VERSION:", tf.__version__)

import tensorflow_addons as tfa
import keras
from tensorflow.keras import layers


class MonetGAN:
    def __init__(self, monet_ds, photo_ds, image_shape):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.strategy = tf.distribute.get_strategy()

        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.image_shape = image_shape

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [*self.image_shape])

        return image

    def read_tfrecord(self, record):
        expected_format = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, expected_format)
        image = self.decode_image(example['image'])

        return image

    def load_dataset(self, dataset):

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=self.AUTOTUNE)

        return dataset

