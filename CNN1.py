#!/usr/bin/env python
# coding: utf-8

# ### Image Classification using Convolutional neural networks
# - We have been tasked to apply Convolutional neural networks to clasify any sample image data of our choice purposely for image recognition
# - We are going to use Covid-19 image Dataset from kaggle to try and detect covid-19 using chest X-rays.
# - Comparisons can be done between normal, covid-19 and pneumonia chest X-rays where relvant and possible
# - Datase from Kaggle : (https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
# - pip install opencv-python
# 

# In[167]:


#suppress warnings for a clean notebook just to moderate error messages
import warnings
warnings.filterwarnings('ignore')


# In[168]:


import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential

import cv2  # Import the OpenCV library
from sklearn.model_selection import train_test_split

#Conv2D for convolutional layer
#Maxpooling2D for pooling layer
#Dense for fully connected layer 
#Flatten to convert or flatten our multiD vectors to a singleD vectors
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Below metrics for testing and validating our models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ### Load Dataset

# In[169]:


X_train = os.listdir(r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\train\Covid")
y_train = os.listdir(r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\train\Normal")

X_test = os.listdir(r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\test\Covid")
y_test=os.listdir(r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\test\Normal")



# ### View items on the directories of interest

# In[170]:


print("Files in the 'Covid' directory:")
for file in X_train:
    print(file)


# In[171]:


print("Files in the 'Normal' directory:")
for file in y_train:
    print(file)


# In[172]:


print("Files in the 'Covid' directory:")
for file in X_test:
    print(file)


# In[173]:


print("Files in the 'Normal' directory:")
for file in y_test:
    print(file)


# In[174]:


X_test[0]


# In[175]:


# Display the image
#plt.imshow(X_test[0])
#plt.title('CNN Image')
#plt.show()


# #### Load and preprocess the data:
# - We will need to load the image data, resize it to a consistent size, and normalize the pixel values. 

# In[177]:


# Initialization of our data of interest

X_train = []
y_train = []

X_test = [] 
y_test = [] 

# Define raw string literals by adding 'r' prefix
train_covid_dir = r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\train\Covid"
train_normal_dir = r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\train\Normal"
test_covid_dir = r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\test\Covid"
test_normal_dir = r"C:\Users\PC\Desktop\KIbe\sem 2\Unstructured data analytics & apps\jupkibe\data\Covid19-dataset\test\Normal"

# Load and preprocess training data
for filename in os.listdir(train_covid_dir):
    img = cv2.imread(os.path.join(train_covid_dir, filename))
    img = cv2.resize(img, (224, 224))  # Resize to a common size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    X_train.append(img)
    y_train.append([1, 0])  # COVID label

# Load and preprocess training data for the Normal class
for filename in os.listdir(train_normal_dir):
    img = cv2.imread(os.path.join(train_normal_dir, filename))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    X_train.append(img)
    y_train.append([0, 1])  # Normal label

# Load and preprocess test data in a similar way
for filename in os.listdir(test_covid_dir):
    img = cv2.imread(os.path.join(test_covid_dir, filename))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    X_test.append(img)
    y_test.append([1, 0])  # COVID label

for filename in os.listdir(test_normal_dir):
    img = cv2.imread(os.path.join(test_normal_dir, filename))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    X_test.append(img)
    y_test.append([0, 1])  # Normal label

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# ### Split the data into training and validation sets:

# In[178]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# ### Here define the CNN model:
# - We now can define a simple CNN architecture using TensorFlow/Keras.

# In[179]:


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 output classes (COVID and Normal)
])


# ### Here we now compile the model:

# In[180]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[181]:


#Data Augmentation although only necessary for small data sets and sure our data set is quite small
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

datagen.fit(X_train)


# ### Here we now  train the model:

# In[182]:


history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,  # You can adjust the number of epochs
    verbose=2
)


# #### Model Evaluation on the test set:

# In[185]:


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')


# In[204]:


# Make predictions on the test data
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

# we have 46 images for normal and covid lets try to detect if those X-rays are either normal or covid infested
# true = 1
# False = 0
# prediction for a specific image (e.g., the first image in the test set)
print("Predictions 0:NO or 1:YES :", predicted_classes[38])


# In[ ]:




