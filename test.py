import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import numpy as np
# use Matplotlib (don't ask)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from IPython.display import Image, SVG
import os
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io

def resize_image(path, path_to):
    i = 0
    for filename in os.listdir(path):
        if filename[-4:] == '.jpg':
            if i < 10000:
                image = io.imread(path + '/' + filename)
                image = color.rgb2gray(image)
                image = resize(image, (128,128))
                io.imsave(path_to + '/' + filename, image)
                i+=1

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

# resize_image('data/test/galaxies', 'data/test_grey/galaxies')
# resize_image('data/train/galaxies', 'data/train_grey/galaxies')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train_grey',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=128,
    class_mode='input')
test_generator = test_datagen.flow_from_directory(
    'data/test_grey',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=128,
    class_mode='input')

x, y = train_generator.next()
print (np.shape(x))
print (np.shape(x[0]))
print (np.shape(x[1]))
print (np.shape(x[2]))

img_shape = np.shape(load_img('data/train_grey/galaxies/100008.jpg'))
img_shape = np.array((128,128,1))
# # Loads the training and test data sets (ignoring class labels)
# (x_train, _), (x_test, _) = mnist.load_data()
# print (np.shape(x_train))
# # Scales the training and test data to range between 0 and 1.
# max_value = float(x_train.max())
# x_train = x_train.astype('float32') / max_value
# x_test = x_test.astype('float32') / max_value
# x_train = x_train.reshape((len(x_train), 28, 28, 1))
# x_test = x_test.reshape((len(x_test), 28, 28, 1))
# print (x_train.shape, x_test.shape)

# set up the autoencoder
autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape = img_shape))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((8, 8, 8)))

# Decoder Layers
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

# setup the encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
# encoder.summary()

batch_size = 128
epochs = 200
# inChannel = 1
# x, y = 128, 128
# input_img = Input(shape = (x, y, inChannel))
#
# autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
# autoencoder.summary()
# autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
autoencoder.load('autoencoder_galaxy_images2.h5')

autoencoder.fit_generator(  train_generator, epochs=epochs, steps_per_epoch = 30,
                            validation_data = (test_generator),
                            validation_steps = 30, use_multiprocessing = True)
# save the weights and the model
autoencoder.save('autoencoder_galaxy_images2.h5')

# show the images at the end
num_images = 10
np.random.seed(42)
x, y = train_generator.next()
encoded_imgs = encoder.predict(x)
decoded_imgs = autoencoder.predict(x)

# print (x[0]/decoded_imgs[0])
# print ()

plt.figure(figsize=(18, 4))
for i in range(0, num_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x[i][:, :, 0], cmap = 'Greys')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[i].reshape(16,32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[i][:, :, 0], cmap = 'Greys')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()



# plt.figure(figsize=(18, 4))

# for i, image_idx in enumerate(random_test_images):
#     # plot original image
#     ax = plt.subplot(3, num_images, i + 1)
#     plt.imshow(x_test[image_idx].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # plot encoded image
#     ax = plt.subplot(3, num_images, num_images + i + 1)
#     plt.imshow(encoded_imgs[image_idx].reshape(16, 8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # plot reconstructed image
#     ax = plt.subplot(3, num_images, 2*num_images + i + 1)
#     plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
