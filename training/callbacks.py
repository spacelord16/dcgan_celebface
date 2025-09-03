import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
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
        noise = tf.random.normal(shape=(n * n, self.latent_dim))
        out = self.model.generator(noise)

        plt.figure(figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                ax = plt.subplot(n, n, k + 1)
                plt.imshow((out[k] + 1) / 2)  # Rescale from [-1, 1] to [0, 1]
                plt.axis("off")
                k += 1

        save_path = os.path.join(
            config.OUTPUT_DIR, f"gen_images_epoch_{epoch+1:04d}.png"
        )
        plt.savefig(save_path)
        plt.close()  # Close the figure to save memory
        print(f"\nSaved generated images to {save_path}")


class GANEarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom early stopping for GANs that monitors both generator and discriminator losses.
    Stops training if:
    1. Either loss becomes NaN or infinite
    2. Generator loss stays too low (discriminator completely fooled)
    3. Discriminator loss stays too low (generator too weak)
    4. Both losses plateau for too long
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        monitor_g_loss_threshold=0.1,
        monitor_d_loss_threshold=0.1,
        verbose=1,
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_g_loss_threshold = monitor_g_loss_threshold
        self.monitor_d_loss_threshold = monitor_d_loss_threshold
        self.verbose = verbose

        # Tracking variables
        self.wait = 0
        self.best_loss_sum = np.inf
        self.g_loss_history = []
        self.d_loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current_g_loss = logs.get("g_loss", 0)
        current_d_loss = logs.get("d_loss", 0)

        # Check for NaN or infinite values
        if (
            np.isnan(current_g_loss)
            or np.isnan(current_d_loss)
            or np.isinf(current_g_loss)
            or np.isinf(current_d_loss)
        ):
            if self.verbose:
                print(
                    f"\nEarly stopping: Loss became NaN or infinite at epoch {epoch + 1}"
                )
            self.model.stop_training = True
            return

        # Track loss history
        self.g_loss_history.append(current_g_loss)
        self.d_loss_history.append(current_d_loss)

        # Check if generator loss is too low (discriminator completely fooled)
        if len(self.g_loss_history) >= 5:
            recent_g_losses = self.g_loss_history[-5:]
            if all(loss < self.monitor_g_loss_threshold for loss in recent_g_losses):
                if self.verbose:
                    print(
                        f"\nEarly stopping: Generator loss too low ({current_g_loss:.4f}) for 5 consecutive epochs"
                    )
                self.model.stop_training = True
                return

        # Check if discriminator loss is too low (generator too weak)
        if len(self.d_loss_history) >= 5:
            recent_d_losses = self.d_loss_history[-5:]
            if all(loss < self.monitor_d_loss_threshold for loss in recent_d_losses):
                if self.verbose:
                    print(
                        f"\nEarly stopping: Discriminator loss too low ({current_d_loss:.4f}) for 5 consecutive epochs"
                    )
                self.model.stop_training = True
                return

        # Monitor combined loss for plateau detection
        current_loss_sum = current_g_loss + current_d_loss

        if current_loss_sum < self.best_loss_sum - self.min_delta:
            self.best_loss_sum = current_loss_sum
            self.wait = 0
            if self.verbose:
                print(
                    f"\nImprovement detected. Best combined loss: {self.best_loss_sum:.4f}"
                )
        else:
            self.wait += 1
            if self.verbose:
                print(
                    f"\nNo improvement for {self.wait} epochs. Combined loss: {current_loss_sum:.4f}"
                )

        if self.wait >= self.patience:
            if self.verbose:
                print(
                    f"\nEarly stopping: No improvement for {self.patience} consecutive epochs"
                )
            self.model.stop_training = True
