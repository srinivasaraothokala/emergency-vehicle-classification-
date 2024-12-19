from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PIL
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tempfile import TemporaryFile

data=pd.read_csv("/content/drive/MyDrive/Emergency_Vehicles/train.csv")

X = []
# iterating over each image
for img_name in data.image_names:
    # loading the image using its name
    img = plt.imread('/content/drive/MyDrive/Emergency_Vehicles/train/' + img_name)
    # normalizing the pixel values
    img = img/255
    # saving each image in the list
    X.append(img)

X=np.array(X)

y = data.emergency_or_not.values

fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
    ax[i].imshow(X[i*400])
    ax[i].axis('off')


X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3, random_state=45)
#shape of traing and validation dataset
(X_train.shape,y_train.shape),(X_valid.shape,y_valid.shape)


final_train=np.array(final_train_data)
final_target_train=np.array(final_target_data)
# shape of new training set
final_train.shape, final_target_train.shape
((5760, 224, 224, 3), (5760,))
# visualizing the augmented images



final_valid = X_valid.reshape(X_valid.shape[0], 224*224*3)
final_valid.shape

np.savez("/content/drive/MyDrive/Emergency_Vehicles/temp1.npz", x=final_train, y=final_target_data)

np.savez("/content/drive/MyDrive/Emergency_Vehicles/temp2.npz", x=final_valid, y=y_valid)


npzfile2 = np.load("/content/drive/MyDrive/Emergency_Vehicles/temp2.npz")

final_train=npzfile2['x']
final_target_train=npzfile2['y']

final_valid=npzfile2['x']
y_valid=npzfile2['y']


adam = tf.keras.optimizers.Adam(lr=1e-5)
# using relu as activation function in hidden layer
model=Sequential()
model.add(layers.InputLayer(input_shape=(224*224*3,)))
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(units=1,activation='sigmoid'))


model.summary()


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_history=model.fit(final_train,final_target_train,epochs=50,batch_size=150,validation_data=(final_valid,y_valid))

model.save('/content/drive/MyDrive/Emergency_Vehicles/emergencyModel.h5')
