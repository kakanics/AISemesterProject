import tensorflow as tf
from tensorflow.keras.utils import plot_model

model = tf.keras.models.load_model("MalariaModel.keras")
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)