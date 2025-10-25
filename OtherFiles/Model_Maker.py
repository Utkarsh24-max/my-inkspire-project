import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

# ============================
# Device Setup
# ============================
if tf.config.list_physical_devices('GPU'):
    print("CUDA is available. Using GPU.")
    device_name = "/GPU:0"
else:
    print("CUDA is not available. Using CPU.")
    device_name = "/CPU:0"
# ============================
# Data Loading
# ============================
imgs_path = glob.glob('/kaggle/input/anime-sketch-colorization-pair/data/train/*.png')
print(f"Number of images: {len(imgs_path)}")

plt.figure(figsize=(12, 8))
for i, img_path in enumerate(imgs_path[:4]):
    img = Image.open(img_path).convert('RGB')
    np_img = np.array(img)
    plt.subplot(2, 2, i + 1)
    plt.imshow(np_img)
    plt.title(str(np_img.shape))
plt.show()
# ============================
# Data Preprocessing
# ============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 256))  # Resize to (512, 256)
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img * 2 - 1  # Scale to [-1, 1]
    w = img.shape[1] // 2  # Split width

    # Extract the sketch (right half) and color (left half)
    color = img[:, :w]
    sketch = img[:, w:]
    # Convert sketch to grayscale
    sketch = np.mean(sketch, axis=-1, keepdims=True)  # Convert to grayscale with shape (256, 256, 1)
    return sketch, color
def data_generator(imgs_path, batch_size):
    while True:
        np.random.shuffle(imgs_path)
        for i in range(0, len(imgs_path), batch_size):
            batch_paths = imgs_path[i:i + batch_size]
            batch_sketches, batch_colors = [], []
            for path in batch_paths:
                sketch, color = preprocess_image(path)
                batch_sketches.append(sketch)
                batch_colors.append(color)
            yield np.array(batch_sketches), np.array(batch_colors)
test = data_generator(imgs_path, 32)
for step, (sketches, real_images) in enumerate(test):
    if (step>19):
        break
    # Plot real vs fake images
    plt.figure(figsize=(12, 6))
    for i in range(4):
        # Plot real images
        plt.subplot(3, 4, i + 1)
        plt.imshow((real_images[i] + 1) / 2)  # Denormalize
        plt.title("Real Image")
        plt.axis('off')

        # Plot input sketches
        plt.subplot(3, 4, i + 9)
        plt.imshow((sketches[i] + 1) / 2)  # Denormalize for RGB sketch
        plt.title("Input Sketch")
        plt.axis('off')

    plt.show()
# ============================
# Model Architectures
# ============================
def build_generator(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    def encoder_block(x, filters, use_batchnorm=True):
        x = layers.Conv2D(filters, (4, 4), strides=2, padding="same")(x)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        return layers.LeakyReLU()(x)

    e1 = encoder_block(inputs, 64, use_batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)

    # Bottleneck
    bottleneck = layers.Conv2D(512, (4, 4), strides=2, padding="same", activation="relu")(e5)

    # Decoder
    def decoder_block(x, skip_input, filters, dropout=False):
        x = layers.Conv2DTranspose(filters, (4, 4), strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if dropout:
            x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        return layers.Concatenate()([x, skip_input])

    d1 = decoder_block(bottleneck, e5, 512, dropout=True)
    d2 = decoder_block(d1, e4, 512, dropout=True)
    d3 = decoder_block(d2, e3, 256)
    d4 = decoder_block(d3, e2, 128)
    d5 = decoder_block(d4, e1, 64)

    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding="same", activation="tanh")(d5)
    return models.Model(inputs, outputs, name="Generator")

def build_discriminator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    def discriminator_block(x, filters, strides=2):
        x = layers.Conv2D(filters, (4, 4), strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.LeakyReLU()(x)

    x = discriminator_block(inputs, 64, strides=2)
    x = discriminator_block(x, 128, strides=2)
    x = discriminator_block(x, 256, strides=2)
    x = discriminator_block(x, 512, strides=1)

    outputs = layers.Conv2D(1, (4, 4), strides=1, padding="same")(x)
    return models.Model(inputs, outputs, name="Discriminator")

# Instantiate models
generator = build_generator(input_shape=(256, 256, 1))
discriminator = build_discriminator(input_shape=(256, 256, 4))  # Input includes sketch + color (concatenated)

# ============================
# Losses and Optimizers
# ============================
cross_entropy = BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    target = tf.cast(target, tf.float32)
    gen_output = tf.cast(gen_output, tf.float32)
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

gen_optimizer = Adam(2e-4, beta_1=0.5)
disc_optimizer = Adam(2e-4, beta_1=0.5)
# ============================
# Training Step
# ============================
@tf.function
def train_step(sketches, colors):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_colors = generator(sketches, training=True)
        # Cast tensors to float32 to ensure consistency
        sketches = tf.cast(sketches, tf.float32)
        colors = tf.cast(colors, tf.float32)

        real_input = tf.concat([sketches, colors], axis=-1)
        generated_input = tf.concat([tf.cast(sketches, tf.float32), tf.cast(generated_colors, tf.float32)], axis=-1)


        disc_real_output = discriminator(real_input, training=True)
        disc_generated_output = discriminator(generated_input, training=True)

        gen_loss = generator_loss(disc_generated_output, generated_colors, colors)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss
def train(data_generator, steps_per_epoch, epochs):
    best_gen_loss = float('inf')  # Initialize with a very high value for comparison
    best_disc_loss = float('inf')  # Track the best discriminator loss (optional)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for step in range(steps_per_epoch):
            sketches, colors = next(data_generator)
            gen_loss, disc_loss = train_step(sketches, colors)
            print(f"Step {step + 1}/{steps_per_epoch}: Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")
            
            # Plot after each step
            # plot_images(sketches, colors)

            # Check if the current generator loss is the best one
            if gen_loss < best_gen_loss:
                best_gen_loss = gen_loss
                # Save the generator model if it improves
                generator.save(f'best_generator.h5')
                print(f"Best Generator model saved as best_generator_epoch_{epoch+1}.h5")

            # Optional: Save the discriminator model if needed
            if disc_loss < best_disc_loss:
                best_disc_loss = disc_loss
                discriminator.save(f'best_discriminator.h5')
                print(f"Best Discriminator model saved as best_discriminator_epoch_{epoch+1}.h5")

        # Plot and save after each epoch
        plot_images(sketches, colors, is_epoch_end=True)

        # Optionally, you can also save the best model at the end of the epoch
        # based on the loss improvement, but the condition above ensures that we only
        # save the model if there is an improvement during the training process.

def plot_images(sketches, colors, is_epoch_end=False):
    sample_sketch = sketches[:4]  # Use the same batch of sketches from the current training step
    sample_real_images = colors[:4]  # Use the same batch of real images
    sample_output = generator(sample_sketch, training=False)  # Generate fake images

    # Cast to float32 before plotting
    sample_real_images = tf.cast(sample_real_images, tf.float32)
    sample_output = tf.cast(sample_output, tf.float32)
    sample_sketch = tf.cast(sample_sketch, tf.float32)

    # Plot real vs fake images
    plt.figure(figsize=(18, 12))
    for i in range(4):
        # Plot real images
        plt.subplot(3, 4, i + 1)
        plt.imshow((sample_real_images[i] + 1) / 2)  # Denormalize
        plt.title("Real Image")
        plt.axis('off')

        # Plot fake images
        plt.subplot(3, 4, i + 5)
        plt.imshow((sample_output[i] + 1) / 2)  # Denormalize
        plt.title("Generated Image")
        plt.axis('off')

        # Plot input sketches
        plt.subplot(3, 4, i + 9)
        plt.imshow((sample_sketch[i] + 1) / 2)  # Denormalize for RGB sketch
        plt.title("Input Sketch")
        plt.axis('off')

    # Show the plot after every step and at the end of the epoch
    if is_epoch_end:
        plt.suptitle(f"Epoch End: Generated vs Real vs Sketches", fontsize=16)
    plt.show()
    plt.close()
batch_size = 32
train_gen = data_generator(imgs_path, batch_size)
steps_per_epoch = len(imgs_path) // batch_size
epochs = 50

train(train_gen, steps_per_epoch, epochs)
generator.save(f'last_generator.h5')
