import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import csv
import time
import os

def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)
    image = tf.image.random_crop(image, size=[200, 200, 1])
    image = tf.image.resize(image, [224, 224])
    return image, label

learning_rates = [0.001, 0.0005]
epsilons = [1e-07, 1e-08]

batch_size = 64

with open('hyperparameter_tuning_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Learning Rate', 'epsilon', 'Accuracy', 'Training Time'])

    for lr in learning_rates:
        for ep in epsilons:
            dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)
            dataset_size = info.splits['train'].num_examples
            sample_size = int(0.1 * dataset_size) 
            train_size = int(0.8 * sample_size)
            val_size = sample_size - train_size

            dataset = dataset.take(sample_size)
            train_dataset = dataset.take(train_size).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = dataset.skip(train_size).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=ep),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            start_time = time.time()
            history = model.fit(train_dataset.batch(32),
                                epochs=50,
                                validation_data=val_dataset.batch(32),
                                verbose=1)
            training_time = time.time() - start_time

            loss, accuracy = model.evaluate(val_dataset.batch(32), verbose=1)
            print(f'Learning Rate: {lr}, epsilon: {ep}, Accuracy: {accuracy:.2f}, Training Time: {training_time:.2f} seconds')

            writer.writerow([lr, batch_size, ep, accuracy, training_time])

            plt.figure()
            plt.plot(history.history['accuracy'], label='accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend(loc='lower right')
            plt.title(f'LR: {lr}, Batch Size: {batch_size}, epsilon: {ep}')
            plt.savefig(f'history_lr{lr}_bs{batch_size}_opt{ep}.png')
            plt.close()
