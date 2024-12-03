import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

modelMM = tf.keras.models.load_model('MalariaModel.keras')
modelRN = tf.keras.models.load_model('resnetFineTuned.keras')
modelVG = tf.keras.models.load_model('vgg16FineTuned.keras')

def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.preprocessing.image.random_shift(image.numpy(), 0.01, 0.01, row_axis=0, col_axis=1, channel_axis=2)
    image = tf.convert_to_tensor(image) 
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_jpeg_quality(image, 97, 100)
    return image, label

dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)
img1 = dataset.skip(1).take(1)
img2 = dataset.take(1)


for image, label in img1:
    x,_ = preprocess_image(image, label)
    plt.imshow(x.numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('picInfPre.png')

for image, label in img2:
    x,_ = preprocess_image(image, label)
    plt.imshow(x.numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('picSimPre.png')