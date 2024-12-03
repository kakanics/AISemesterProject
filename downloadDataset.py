import tensorflow_datasets as tfds

builder = tfds.builder('malaria', data_dir='dataset')
builder.download_and_prepare()
print(f"Dataset saved in: {builder.data_dir}")
dataset = builder.as_dataset()

info = builder.info
print(info)