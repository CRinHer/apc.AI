import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback

# Define the callback to print loss
def print_loss(epoch, logs):
    print(f"Epoch: {epoch+1}, Loss: {logs['loss']}")

# Prepare the dataset
image_dir = '../generate/preprocessing/Images'  # Replace with actual path
print("Compiling dataset...")
dataset = tf.keras.utils.image_dataset_from_directory(
    directory=image_dir,
    labels=None,
    color_mode='grayscale',
    image_size=(200, 200)
)
print("Dataset Compiled!")

# Split the dataset into training and validation sets
dataset_size = len(dataset)  # Get the total size of the dataset
train_size = int(0.8 * dataset_size)  # 80% for training
val_size = dataset_size - train_size  # Remaining for validation

# Create training and validation datasets
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)

# Convert the batches into numpy arrays
print("Converting training dataset into numpy array...")
x_train = np.concatenate([batch.numpy() for batch in train_dataset], axis=0).astype('float32') / 255.
print(f"Training dataset shape: {x_train.shape}")

print("Converting validation dataset into numpy array...")
x_test = np.concatenate([batch.numpy() for batch in val_dataset], axis=0).astype('float32') / 255.
print(f"Validation dataset shape: {x_test.shape}")

# Reshape the arrays into the correct shape (1500, 200, 200, 1)
print("Reshaping training data...")
x_train = np.reshape(x_train, (-1, 200, 200, 1))
print(f"Reshaped training dataset shape: {x_train.shape}")

print("Reshaping validation data...")
x_test = np.reshape(x_test, (-1, 200, 200, 1))
print(f"Reshaped validation dataset shape: {x_test.shape}")

# Define the input shape
input_img = keras.Input(shape=(200, 200, 1))

# Encoder
print("Building encoder layers...")
x = layers.Conv2D(40, (20, 20), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((4, 4), padding='same')(x)  # 50x50
x = layers.Conv2D(20, (2, 2), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 25x25
encoded = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x)

# Decoder
print("Building decoder layers...")
x = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)  # 50x50
x = layers.Conv2D(20, (2, 2), activation='relu', padding='same')(x)
x = layers.UpSampling2D((4, 4))(x)  # 200x200
decoded = layers.Conv2D(1, (20, 20), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
print("Creating the autoencoder model...")
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print("Model compiled.")

# Train the autoencoder with a loss callback
print("Training the model...")
loss_callback = LambdaCallback(on_epoch_end=print_loss)
autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=200,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[loss_callback])

# Save the model's weights to a file
print("Saving the model's weights...")
autoencoder.save_weights('autoencoder_weights.h5')

# Later, you can load the weights back using:
# autoencoder.load_weights('autoencoder_weights.h5')

# Obtain the final weights of the trained model
print("Getting the final weights...")
weights = autoencoder.get_weights()

# Print the shapes of the weights of each layer
print("Printing weight shapes for each layer...")
for i, weight in enumerate(weights):
    print(f"Layer {i} weights: {weight.shape}")

# Predict the decoded images using the trained model
print("Predicting with the trained model...")
decoded_imgs = autoencoder.predict(x_test)
print("Prediction complete.")