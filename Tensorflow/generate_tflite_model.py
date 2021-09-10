#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

### Fetch mnist dataset ###
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
	(tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
	tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

### Build a model ###
tf_model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(10, 28*28, 1)),
	tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(10)
])

### Train the model ###
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_step(images, labels):
	with tf.GradientTape() as tape:
		logits = tf_model(images, training=True)

		loss_value = loss_object(labels, logits)

	grads = tape.gradient(loss_value, tf_model.trainable_variables)
	optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

for (batch, (images, labels)) in enumerate(dataset):
	train_step(images, labels)

### Save the tf model ###
tf.saved_model.save(tf_model, 'example_model')

### Convert it to tf-lite ###
converter = tf.lite.TFLiteConverter.from_saved_model('example_model')
tflite_model = converter.convert()

### Save the tf-lite model ###
with open('example_model.tflite', 'wb') as f:
	f.write(tflite_model)
