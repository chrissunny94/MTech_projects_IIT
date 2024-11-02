# Import necessary libraries
#!pip install -r requirement.txt
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, Reshape, ReLU
from keras.preprocessing.image import ImageDataGenerator

sns.set(style="darkgrid", color_codes=True)

# Load and preprocess the dataset
img_width, img_height = 256, 256
batchsize = 32

train = keras.utils.image_dataset_from_directory(
    directory='animefacedataset',
    batch_size=batchsize,
    image_size=(img_width, img_height))

data_iterator = train.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))

DIR = 'animefacedataset'
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    DIR,
    target_size=(64, 64),
    batch_size=batchsize,
    class_mode=None)

# Create the Generator model
KI = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
input_dim = 300

def Generator_Model():
    Generator = Sequential()
    Generator.add(Dense(8 * 8 * 512, input_dim=input_dim))
    Generator.add(ReLU())
    Generator.add(Reshape((8, 8, 512)))
    Generator.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=KI, activation='ReLU'))
    Generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=KI, activation='ReLU'))
    Generator.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=KI, activation='ReLU'))
    Generator.add(Conv2D(3, (4, 4), padding='same', activation='sigmoid'))
    return Generator

generator = Generator_Model()
generator.summary()

# Create the Discriminator model
def Discriminator_Model():
    input_shape = (64, 64, 3)
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(3, 3), activation='LeakyReLU', input_shape=input_shape))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(Conv2D(128, kernel_size=(3, 3), activation='LeakyReLU'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(Conv2D(256, kernel_size=(3, 3), activation='LeakyReLU'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(Flatten())
    discriminator.add(Dense(256, activation='LeakyReLU'))
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator

discriminator = Discriminator_Model()
discriminator.summary()

# Create the GAN model
def GAN_Model(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = GAN_Model(generator, discriminator)

# Train the GAN model
def train_gan(gan, generator, discriminator, dataset, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 300))
        generated_images = generator.predict(noise)
        
        real_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
        X = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9  # Label smoothing
        
        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)
        
        noise = np.random.normal(0, 1, (batch_size, 300))
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)
        
        if epoch % 1000 == 0:
            print('Epoch: {}'.format(epoch))


train_gan(gan, generator, discriminator, train_generator)

# Evaluate model results
def plot_generated_images(generator, n=10):
    noise = np.random.normal(0, 1, (n, 300))
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.show()

plot_generated_images(generator)
