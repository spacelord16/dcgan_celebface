import tensorflow as tf
import matplotlib.pyplot as plt
import os
import config

class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        # Ensure the output directory exists
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)

    def on_epoch_end(self, epoch, logs=None):
        n = 6
        k = 0
        noise = tf.random.normal(shape=(n*n, self.latent_dim))
        out = self.model.generator(noise)
        
        plt.figure(figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                ax = plt.subplot(n, n, k + 1)
                plt.imshow((out[k] + 1) / 2) # Rescale from [-1, 1] to [0, 1]
                plt.axis('off')
                k += 1
        
        save_path = os.path.join(config.OUTPUT_DIR, f"gen_images_epoch_{epoch+1:04d}.png")
        plt.savefig(save_path)
        plt.close() # Close the figure to save memory
        print(f"\nSaved generated images to {save_path}")