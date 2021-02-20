# import library
from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import lite
from tensorflow.keras import models

# defind variable
feature_sets_filename = 'all_targets_mfcc_sets.npz'
model_filename = 'mfcc_16_16.h5'
checkpoint_path = 'checkpoint/weights.best.hdf5'

# Load feature sets
feature_sets = np.load(feature_sets_filename)

# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

y_train = to_categorical(y_train, num_classes=30)
y_val = to_categorical(y_val, num_classes=30)
y_test = to_categorical(y_test, num_classes=30)


# CNN for TF expects (batch, height, width, channels)
# So we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)

# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]

loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

# Build model
# Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
model = models.Sequential()
model.add(layers.Conv2D(32,
                        (2, 2),
                        activation='relu',
                        input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(30, activation='softmax'))

# Display model
model.summary()

# Add training parameters to model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

ck = ModelCheckpoint(checkpoint_path, monitor='val_acc',
                     verbose=1, save_best_only=True, mode='max')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)

# Train
history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[ck, es])

# Plot results

acc = history.history['acc']
# acc = history.history['categorical_accuracy']
val_acc = history.history['val_acc']
# val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Save the model as a file
models.save_model(model, model_filename)

labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine',
          'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']


predictions = model.predict(x_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labels))

# Evaluate model with test set
model.evaluate(x=x_test, y=y_test)

# Parameters
keras_model_filename = 'mfcc_16_16.h5'
tflite_filename = 'mfcc_16_16.tflite'


model = models.load_model(keras_model_filename)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)

