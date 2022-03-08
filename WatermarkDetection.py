#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


raw_dir = 'C:/Users/Administrator/Downloads/dk-dataset/dataset/train/negative'
wat_dir = 'C:/Users/Administrator/Downloads/dk-dataset/dataset/train/positive'

raw_datagen = ImageDataGenerator(rescale=1./255)

raw_generator = raw_datagen.flow_from_directory(
        raw_dir,  # this is the target directory
        target_size=(250, 250),
        batch_size=5096, shuffle = False,
        class_mode="categorical")

wat_datagen = ImageDataGenerator(rescale=1./255)

wat_generator = wat_datagen.flow_from_directory(
        wat_dir,  # this is the target directory
        target_size=(250, 250),
        batch_size=5096, shuffle = 0,
        class_mode="categorical")

x_raw,y = raw_generator.next()
x_wat,y = wat_generator.next()


# In[5]:


test_dir = 'C:/Users/Administrator/Downloads/dk-dataset/dataset/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        target_size=(250, 250),
        batch_size=1200, shuffle = False,
        class_mode="categorical")

x_test,y_test = test_generator.next()


# In[18]:


x_train_raw = x_raw[0:3900]
x_train_wat = x_wat[0:3900]

x_val_raw = x_raw[3900:4200]
x_val_wat = x_wat[3900:4200]

x_train = np.vstack((x_train_raw, x_train_wat))
x_val = np.vstack((x_val_raw, x_val_wat))

y_train = np.repeat([0.], 3900)
y_train = np.append(y_train, np.repeat([1.], 3900))

y_val = np.repeat([0.], 300)
y_val = np.append(y_val, np.repeat([1.], 300))


# In[23]:


x_val.shape


# In[24]:


y_val.shape


# In[25]:


model = keras.models.Sequential()
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='elu', input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='elu'))
model.add(layers.Conv2D(100, (3, 3), activation='elu'))
model.add(layers.MaxPooling2D((4, 4)))

model.add(layers.Conv2D(75, (3, 3), activation='elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(202, activation='elu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(40, activation='elu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
#model.summary()
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 25, 
          batch_size = 30, verbose = 0)


# In[26]:


# make a prediction
yhat = model.predict(x_test[0:])


# In[27]:


print(yhat[0])


# In[28]:


def run_fast_scandir(r_path):
    subfolders, files = [], []
    for f in os.scandir(r_path):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            files.append(f.name)
            
    return files


# In[29]:


Pic_names=run_fast_scandir('C:/Users/Administrator/Downloads/dk-dataset/dataset/test/test')


# In[30]:


(Pic_names)[100]


# In[31]:


import csv  

header = ['name','predicted']


with open('output.csv', 'w', newline="") as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    for i in range(len(Pic_names)):
        y=0
        if yhat[i]>=0.5:
          y=1
        else:
          y=0
        data=[Pic_names[i],y]
        writer.writerow(data)


# In[ ]:




