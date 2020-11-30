import sys
sys.path.append('./GAN/')
from MonetGAN import MonetGAN

'''
The purpose of this code is to reproduce images from the photo_images list in the 'style' of Monet, using the images
in the monet_images list as training data. You can sample some of the images below to get a sense of what everything
looks like.
'''

monet_path = './dataset/monet_jpg/*.jpg'
photo_path = './dataset/photo_jpg/*.jpg'
image_shape = (256, 256, 3)

GAN = MonetGAN(monet_path, photo_path, image_shape, batch_size=1)

# example_monet = GAN.load_monet(index=0)
example_photo = GAN.load_photo(index=0)
generated_image = GAN.generator.generate(example_photo)
print(generated_image)
print(generated_image.shape)
print(type(generated_image))

GAN.display_image(generated_image, save=True)

# GAN.display_image(example_monet)
# GAN.display_image(example_photo)

# GAN.train(steps=10, batch_size=2)

