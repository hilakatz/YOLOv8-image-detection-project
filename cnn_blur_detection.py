import os
from os import listdir
import random
import shutil

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import cv2

#source: https://www.kaggle.com/code/ahmetfurkandemr/blur-detection-with-tflite

path = "/content/drive/MyDrive/Image processing projects/kaggle/blur-dataset"

listImage = []

for i in os.listdir(path):
  if i=="sharp" or i=="blur_dataset_scaled":
    continue
  for j in os.listdir(path+"/"+i):
    try:
      if j.split(".")[1]!="JPG" and j.split(".")[1]!="jpg" and j.split(".")[1]!="jpeg":
        pass
    except:
      pass
    listImage.append(path+"/"+i+"/"+j)

random.shuffle(listImage)
train_blur = listImage[0:500]
test_blur = listImage[600:]

print(len(train_blur))
print(len(test_blur))

for i in train_blur:
  image_name = i.split("/")
  image_name = image_name[len(image_name)-1]

  try:
    shutil.copyfile(i, "/content/drive/MyDrive/Image processing projects/kaggle/train/blurImage/"+image_name)
  except:
    pass

len(os.listdir("/content/drive/MyDrive/Image processing projects/kaggle/train/blurImage/"))

for i in test_blur:
  image_name = i.split("/")
  image_name = image_name[len(image_name)-1]
  shutil.copyfile(i, "/content/drive/MyDrive/Image processing projects/kaggle/test/blurImage/"+image_name)

  len(os.listdir("/content/drive/MyDrive/Image processing projects/kaggle/test/blurImage/"))

  listImage = []

for i in os.listdir(path):
    if i != "sharp":
        continue
    for j in os.listdir(path + "/" + i):
        try:
            if j.split(".")[1] != "JPG" and j.split(".")[1] != "jpg" and j.split(".")[1] != "jpeg":
                pass
        except:
            pass
        listImage.append(path + "/" + i + "/" + j)

random.shuffle(listImage)
train_sharp = listImage[0:300]
test_sharp = listImage[300:]
print(len(train_sharp))
print(len(test_sharp))

for i in train_sharp:
    image_name = i.split("/")
    image_name = image_name[len(image_name) - 1]
    try:
        shutil.copyfile(i, "/content/drive/MyDrive/Image processing projects/kaggle/train/sharpImage/" + image_name)
    except:
        pass

len(os.listdir("/content/drive/MyDrive/Image processing projects/kaggle/train/sharpImage/"))

for i in test_sharp:
    image_name = i.split("/")
    image_name = image_name[len(image_name) - 1]
    try:
        shutil.copyfile(i,"/content/drive/MyDrive/Image processing projects/kaggle/test/sharpImage/" + image_name)
    except:
        pass

len(os.listdir("/content/drive/MyDrive/Image processing projects/kaggle/test/sharpImage/"))

IMAGE_SIZE = 600
BATCH_SIZE = 32
path = "/content/drive/MyDrive/Image processing projects/kaggle"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    path+"/train",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

val_generator = datagen.flow_from_directory(
    path+"/test",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

image_batch, label_batch = next(val_generator)
image_batch.shape, label_batch.shape

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

model.save('cnn_blur_model.keras')

# to load the model: new_model = tf.keras.models.load_model('cnn_blur_model.keras')
