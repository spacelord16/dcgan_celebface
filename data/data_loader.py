import tensorflow as tf
import config

def preprocess_image(image):
  """Normalizes images to the range [-1, 1]."""
  return tf.cast(image, tf.float32) / 127.5 - 1.0

def get_dataset():
  """Loads, preprocesses, and prepares the CelebA dataset pipeline."""
  dataset = tf.keras.preprocessing.image_dataset_from_directory(
      config.DATASET_PATH, 
      label_mode=None, 
      image_size=(config.IM_SHAPE[0], config.IM_SHAPE[1]), 
      batch_size=config.BATCH_SIZE
  )
  
  train_dataset = (
      dataset
      .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
      .shuffle(buffer_size=1024)
      .batch(config.BATCH_SIZE, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
  )
  
  print("✅ Dataset loaded and preprocessed.")
  return train_dataset