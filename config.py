# Data parameters
DATASET_PATH = "/content/dataset/img_align_celeba/img_align_celeba"
IM_SHAPE = (64, 64, 3)
BATCH_SIZE = 128

# Model parameters
LATENT_DIM = 100

# Training parameters
EPOCHS = 20 # Or 1000 as you had in the fit call
LEARNING_RATE = 2e-4
BETA_1 = 0.5 # Adam optimizer beta_1

# Output parameters
OUTPUT_DIR = "generated"