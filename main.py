from __future__ import unicode_literals
import string
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pathlib
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools


_URL = 'file:///' + str(Path(pathlib.Path.cwd(), 'glasses_and_faces.zip'))
zip_dir = tf.keras.utils.get_file('glasses_and_faces.zip', origin=_URL, extract=True)

base_dir = os.path.dirname(zip_dir)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_glasses_dir = os.path.join(train_dir, 'glasses')
train_faces_dir = os.path.join(train_dir, 'faces')

validation_glasses_dir = os.path.join(validation_dir, 'glasses')
validation_faces_dir = os.path.join(validation_dir, 'faces')


num_glasses_tr = len(os.listdir(train_glasses_dir))
num_faces_tr = len(os.listdir(train_faces_dir))

num_glasses_val = len(os.listdir(validation_glasses_dir))
num_faces_val = len(os.listdir(validation_faces_dir))

total_train = num_glasses_tr + num_faces_tr
total_val = num_glasses_val + num_faces_val

print('Людей с очками в тестовом наборе данных: ', num_glasses_tr)
print('Людей без очков в тестовом наборе данных: ', num_faces_tr)

print('Людей с очками в валидационном наборе данных: ', num_glasses_val)
print('Людей без очков в валидационном наборе данных: ', num_faces_val)
print('--')
print('Всего изображений в тренировочном наборе данных: ', total_train)
print('Всего изображений в валидационном наборе данных: ', total_val)


BATCH_SIZE = 100
IMG_SHAPE = 150

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 20
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.legend(loc='lower right')
plt.title('Точность на обучающих и валидационных данных')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери на обучающих и валидационных данных')
plt.savefig('./foo.png')
plt.show()

model_json = model.to_json()
json_file = open('model_json.json'.format(string), 'w')
json_file.write(model_json)
json_file.close()

onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, os.path.join(Path(pathlib.Path.cwd()), 'model.onnx'))
model.save_weights(os.path.join(Path(pathlib.Path.cwd()), 'saved_model.h5'))

