from tensorflow.keras import layers, Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import preprocess_data, visualize_images
import tensorflow as tf

# Mount Google Drive if necessary
# from google.colab import drive
# drive.mount('/content/drive')

# Load and preprocess the data
data = pd.read_csv("data/train.csv")
train_path = "/content/drive/MyDrive/Emergency_Vehicles/train/"

X, y = preprocess_data(data, train_path)

# Visualize sample images
visualize_images(X, 5)

# Split into training and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=45)
X_train = X_train.reshape(X_train.shape[0], 224*224*3)
X_valid = X_valid.reshape(X_valid.shape[0], 224*224*3)

# Save datasets to compressed files
np.savez("data/temp1.npz", x=X_train, y=y_train)
np.savez("data/temp2.npz", x=X_valid, y=y_valid)

# Load data from saved .npz files (example)
npzfile_train = np.load("data/temp1.npz")
npzfile_valid = np.load("data/temp2.npz")
final_train = npzfile_train['x']
final_target_train = npzfile_train['y']
final_valid = npzfile_valid['x']
y_valid = npzfile_valid['y']

# Define the model
model = Sequential([
    layers.InputLayer(input_shape=(224*224*3,)),
    layers.Dense(100, activation='sigmoid'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.5),
    layers.Dense(100, activation='sigmoid'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.4),
    layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])
model.summary()

# Train the model
model_history = model.fit(final_train, final_target_train, epochs=50, batch_size=150, validation_data=(final_valid, y_valid))

# Save the trained model
model.save("models/emergencyModel.h5")
