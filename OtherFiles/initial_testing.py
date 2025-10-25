import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to preprocess the standalone sketch input
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to match generator input dimensions
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img * 2 - 1  # Scale to [-1, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to generate a colorized image from the sketch
def generate_image(model, sketch_image):
    generated_image = model.predict(sketch_image)
    generated_image = (generated_image[0] + 1) / 2.0  # Denormalize to [0, 1]
    generated_image = (generated_image * 255.0).astype(np.uint8)  # Scale to [0, 255]
    return generated_image

# Function to display the outputs of all generators
def display_all_outputs(sketch_image, models, model_names):
    num_models = len(models)
    plt.figure(figsize=(15, 3))  # Adjust figure size for a single-row layout

    # Display input sketch
    plt.subplot(1, num_models + 1, 1)
    plt.imshow(sketch_image[0, :, :, 0], cmap='gray')
    plt.title("Input Sketch")
    plt.axis('off')

    # Display outputs from each generator
    for i, (model, name) in enumerate(zip(models, model_names)):
        generated_image = generate_image(model, sketch_image)
        plt.subplot(1, num_models + 1, i + 2)
        plt.imshow(generated_image)
        plt.title(f"{name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Path to the standalone sketch
    sketch_path = '../saved_images/1.png'  # Replace with your sketch file path

    # Load and preprocess the sketch
    sketch_image = load_and_preprocess_image(sketch_path)

    # List of generator model filenames
    model_filenames = ['../Models/generator1.h5', '../Models/generator2.h5', '../Models/generator3.h5', '../Models/generator4.h5', '../Models/generator5.h5']
    model_names = [f"Generator {i+1}" for i in range(len(model_filenames)-1)]
    model_names.append('Last Generator')

    # Load all models
    models = [tf.keras.models.load_model(filename) for filename in model_filenames]

    # Display all outputs
    display_all_outputs(sketch_image, models, model_names)

if __name__ == "__main__":
    main()
