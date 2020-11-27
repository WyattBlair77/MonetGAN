import sys
sys.path.append('./Our GAN/')

import tensorflow as tf
import os
from MonetGAN import MonetGAN

print("VERSION:", tf.__version__)

'''
The purpose of this code is to reproduce images from the photo_images list in the 'style' of Monet, using the images
in the monet_images list as training data. You can sample some of the images below to get a sense of what everything
looks like.
'''

monet_dataset = [tf.data.TFRecordDataset('./monet_tfrec/'+f) for f in os.listdir('./monet_tfrec/')]
photo_dataset = [tf.data.TFRecordDataset('./photo_tfrec/'+f) for f in os.listdir('./photo_tfrec/')]

image_shape = (256, 256, 3)

MonetGAN = MonetGAN(monet_dataset, photo_dataset, image_shape)
