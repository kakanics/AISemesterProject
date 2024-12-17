import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = tf.image.rgb_to_grayscale(image)
    image = tf.repeat(image, 3, axis=-1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)
    return image, label

batch_size = 64
learning_rate = 0.001
epsilon = 1e-07

dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)
dataset_size = info.splits['train'].num_examples
sample_size = int(0.2 * dataset_size)  
train_size = int(0.8 * sample_size)

val_size = sample_size - train_size
dataset = dataset.take(sample_size)
train_dataset = dataset.take(train_size).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
              loss='binary_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
history = model.fit(train_dataset.batch(batch_size),
                    epochs=50, 
                    validation_data=val_dataset.batch(batch_size),
                    verbose=1)
training_time = time.time() - start_time
model.save('resnetFineTuned.keras')

loss, accuracy = model.evaluate(val_dataset.batch(batch_size), verbose=1)
print(f'Accuracy: {accuracy:.2f}, Training Time: {training_time:.2f} seconds')
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training History')
plt.savefig('history_resnetFineTuned.png')
plt.close()
