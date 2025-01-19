import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib as plt


# Define the input shape
input_img = keras.Input(shape=(200, 200, 1))

# Encoder
x = layers.Conv2D(40, (20, 20), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((4, 4), padding='same')(x) # 50x50
x = layers.Conv2D(20, (2, 2), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x) # 25x25
encoded = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x)

# Decoder
x = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)  # 50x50
x = layers.Conv2D(20, (2, 2), activation='relu', padding='same')(x)
x = layers.UpSampling2D((4, 4))(x)  # 200x200
decoded = layers.Conv2D(1, (20, 20), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# Use the trained autoencoder
encoded_imgs = autoencoder.predict(x_test)
decoded_imgs = autoencoder.predict(encoded_imgs)