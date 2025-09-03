import tensorflow as tf
import matplotlib.pyplot as plt

# Import from our modules
import config
from data.data_loader import get_dataset
from models.generator import build_generator
from models.discriminator import build_discriminator
from training.gan_model import GAN
from training.callbacks import ShowImage, GANEarlyStopping


def main():
    print("Starting DCGAN Training...")

    # 1. Load Data
    train_dataset = get_dataset()

    # 2. Build Models
    discriminator = build_discriminator()
    generator = build_generator()

    # 3. Instantiate and Compile the GAN
    gan = GAN(discriminator, generator)

    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1
    )
    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    gan.compile(d_optimizer, g_optimizer, loss_fn)

    # 4. Set up Callbacks and Train
    image_callback = ShowImage(config.LATENT_DIM)
    early_stopping = GANEarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        monitor_g_loss_threshold=config.EARLY_STOPPING_G_LOSS_THRESHOLD,
        monitor_d_loss_threshold=config.EARLY_STOPPING_D_LOSS_THRESHOLD,
        verbose=1,
    )

    print("Beginning training...")
    history = gan.fit(
        train_dataset, epochs=config.EPOCHS, callbacks=[image_callback, early_stopping]
    )

    print("Training finished!")

    # Optional: Plot and save the final loss history
    plt.plot(history.history["d_loss"], label="d_loss")
    plt.plot(history.history["g_loss"], label="g_loss")
    plt.title("GAN Loss During Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("gan_loss_history.png")
    plt.show()


if __name__ == "__main__":
    main()
