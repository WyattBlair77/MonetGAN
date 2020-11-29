import tensorflow as tf
print("VERSION:", tf.__version__)

from Discriminator import Discriminator
from Generator import Generator
import numpy as np
import glob
import math
import cv2


class MonetGAN:
    def __init__(self, monet_path, photo_path, image_shape):

        self.monet_path = monet_path
        self.photo_path = photo_path
        self.monets = glob.glob(self.monet_path)
        self.photos = glob.glob(self.photo_path)
        self.num_monets = len(self.monets)
        self.num_photos = len(self.photos)

        self.image_shape = image_shape

        self.generator = Generator(self. monets, self.photos, self.image_shape).build()
        self.discriminator = Discriminator(self. monets, self.photos, self.image_shape).build()

    def load_monet(self, index):
        if index >= self.num_monets:
            index = index % self.num_monets

        monet = cv2.imread(self.monets[index])
        return monet

    def load_photo(self, index):
        if index >= self.num_photos:
            index = index % self.num_photos

        photo = cv2.imread(self.photos[index])
        return photo

    def train(self, steps, batch_size=32):

        for step in range(steps):

            # Determine proportion of batch that will be real vs. fake and from where in Monets we will sample
            index = np.random.randint(0, self.num_monets, batch_size)
            num_fake_images = math.floor(np.random.random()*batch_size)
            num_real_images = batch_size - num_fake_images

            # Create the training batch
            discriminator_train = []
            truth_tracker = []

            # Fake images = generated images, truth value is 0
            for n in range(num_fake_images):
                photo = self.load_photo(np.random.choice(self.photos))
                discriminator_train.append(self.generator.generate(photo))
                truth_tracker.append(0)

            # Real images = Monet images, truth value is 1
            for n in range(num_real_images):
                discriminator_train.append(self.load_monet(self.monets[index+n]))
                truth_tracker.append(1)

            # Shuffle the batch
            temp = list(zip(discriminator_train, truth_tracker))
            np.random.shuffle(temp)
            discriminator_train, truth_tracker = zip(*temp)
            discriminator_train = list(discriminator_train)
            truth_tracker = list(truth_tracker)




