import tensorflow as tf
print("VERSION:", tf.__version__)

from Discriminator import Discriminator
from Generator import Generator
import numpy as np
import math
import glob
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image


class MonetGAN:
    def __init__(self, monet_path, photo_path, image_shape, batch_size):

        # Load Datasets
        self.monet_path = monet_path
        self.photo_path = photo_path
        self.monets = glob.glob(self.monet_path)
        self.photos = glob.glob(self.photo_path)
        self.num_monets = len(self.monets)
        self.num_photos = len(self.photos)
        self.batch_size = batch_size

        # Data shape
        self.image_shape = image_shape

        # Instantiate models
        self.generator = Generator(self. monets, self.photos, self.image_shape, self.batch_size)
        self.generator.build()
        self.generator.model.compile(loss='binary_crossentropy')

        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='*3)

        self.discriminator = Discriminator(self. monets, self.photos, self.image_shape, self.batch_size)
        self.discriminator.build()
        self.discriminator.model.compile(loss='binary_crossentropy')

    def save_model(self, model_name):
        path = '../SavedModels/'+model_name+'/'

        os.mkdir(path)
        self.generator.model.save_weights(path+'generator')
        self.discriminator.model.save_weights(path+'discriminator')

    def display_image(self, image, save=False, save_path='./SavedOutput/img.png'):
        img = Image.fromarray(image, 'RGB')
        if save:
            img.save(save_path)
        img.show()

    def load_image(self, path):
        i = Image.open(path)
        x = np.asarray(i, dtype='float32')
        return x

    # Function to load the nth monet painting (cycles if n >= len(monets))
    def load_monet(self, index):
        if index >= self.num_monets:
            index = index % self.num_monets

        monet = self.load_image(self.monets[index])
        return monet

    # Function to load the nth photo (cycles if n >= len(photos))
    def load_photo(self, index):
        if index >= self.num_photos:
            index = index % self.num_photos

        photo = self.load_image(self.photos[index])
        return photo

    # THE training loop:
    # 1) Create batch of training data for the discriminator including some generated images
    # 2) Have the discriminator assess the batch and return its scores
    # 3) Grade the discriminator on how well it identified Monets and grade the Generator on how well it tricked the
    #    discriminator
    # 4) Back-propagate and repeat until we've gone through all the "steps" (parameter)
    def train(self, steps):

        valid = np.ones(self.batch_size)

        # Training loop
        for step in range(steps):

            # Determine proportion of batch that will be real vs. fake and from where in Monets we will sample
            index = np.random.randint(0, self.num_monets)
            num_fake_images = math.floor(np.random.random()*self.batch_size)
            num_real_images = self.batch_size - num_fake_images

            # Create the training batch
            discriminator_train, generator_train = [], []
            truth_tracker = []

            # Fake images = generated images, truth value is 0
            for n in range(num_fake_images):

                photo = self.load_photo(index+n)
                generator_train.append(photo)

                discriminator_train.append(self.generator.generate(photo))
                truth_tracker.append(0)

            # Real images = Monet images, truth value is 1
            for n in range(num_real_images):
                discriminator_train.append(self.load_monet(index+n))
                truth_tracker.append(1)

            # Shuffle the batch
            temp = list(zip(discriminator_train, truth_tracker))
            np.random.shuffle(temp)
            discriminator_train, truth_tracker = zip(*temp)
            discriminator_train = list(discriminator_train)
            truth_tracker = list(truth_tracker)

            # Train discriminator
            d_loss = self.discriminator.model.train_on_batch(discriminator_train, truth_tracker)
            g_loss = self.generator.model.train_on_batch(generator_train, valid)

            print("%d [D loss: %f] [G loss: %f]" % (step, 100 * d_loss, g_loss))



