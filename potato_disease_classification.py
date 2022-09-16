#!/usr/bin/env python
# coding: utf-8

# In[15]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


# In[45]:


IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 15


# In[31]:


train = ImageDataGenerator(rescale = 1./255)
validation = ImageDataGenerator(rescale = 1./255)


# In[32]:


TRAIN_PATH = "D:/potato_disease_classification/datasplitted/train"
VAL_PATH = "D:/potato_disease_classification/datasplitted/val"


# In[33]:


train_data = train.flow_from_directory(TRAIN_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="categorical",
                                       seed=2022)


# In[34]:


val_data = validation.flow_from_directory(VAL_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="categorical",
                                       seed=2022)


# In[35]:


train_data.class_indices


# In[46]:


model = Sequential(
    [
        Conv2D(32,(3,3), activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'), 
        Dense(3, activation='softmax')  
    ]
)


# In[38]:


model.summary()


# In[47]:


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[48]:


history = model.fit(train_data,
                    steps_per_epoch = 1506/BATCH_SIZE,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_data
                   )


# In[49]:


history.history.keys()


# In[50]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[52]:


import matplotlib.pyplot as plt
plt.plot(range(EPOCHS), acc, label="Training Accuracy")
plt.plot(range(EPOCHS), val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Training and Validaiton Accuracy")
plt.show()


# In[53]:


plt.plot(range(EPOCHS), loss, label="Training Loss")
plt.plot(range(EPOCHS), val_loss, label="Validation Loss")
plt.legend()
plt.title("Training and Validaiton Loss")
plt.show()


# In[55]:


version=1
model.save(f"D:/potato_disease_classification/model/{version}")

