from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)
c = 0

# NOTE: The model path 'Models/generator5.h5' must be correct and the file must be present
# in your Hugging Face Space repository.
try:
    # Set logging level to suppress detailed warnings during load
    tf.get_logger().setLevel('ERROR') 
    model = tf.keras.models.load_model('Models/generator5.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a placeholder function if model fails to load to prevent startup crash
    def placeholder_predict(sketch):
        print("Using placeholder prediction (Model failed to load).")
        # Return a simple grayscale image array to match expected output shape
        return np.zeros((1, 256, 256, 3), dtype=np.float32) 
    model = type('DummyModel', (object,), {'predict': placeholder_predict})()


def preprocess_image(image_data):
    """
    Convert base64 image to a NumPy array and preprocess for the model.
    Adds batch and channel dimensions to the input.
    """
    # Decode base64 image
    # Note: Splitting by comma handles the "data:image/png;base64," prefix
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("RGBA")
    
    # Add white background if image has transparency
    if image.mode == 'RGBA':
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert("L")
    else:
        image = image.convert("L")
    
    image = image.resize((256, 256))  # Resize to match the model input size
    # Check if image is truly grayscale (1 channel)
    image_array = np.array(image)
    if image_array.ndim == 3:
        # If it somehow ended up RGB, convert it back to single channel L
        image_array = image_array[..., 0] 
        
    image_array = image_array / 255.0  # Normalize to [0, 1]
    
    # Add batch and channel dimensions
    # Shape: (1, 256, 256, 1) for grayscale input
    image_array = image_array[np.newaxis, ..., np.newaxis] 
    return image_array

def postprocess_image(model_output):
    """
    Convert the model output (expected to be 3-channel, range [-1, 1]) to a base64-encoded PNG image.
    """
    # Model output is expected to be normalized in [-1, 1], so rescale it to [0, 255]
    # The output is expected to be (1, 256, 256, 3) for an RGB image.
    image = ((model_output[0] + 1) * 127.5).astype(np.uint8)
    
    # If the model output is not 3 channels, handle it (e.g., if it's 1-channel grayscale output)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1) # Stack to create RGB if it's grayscale

    image = Image.fromarray(image)
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_image(image, prefix='image'):
    """
    Placeholder for saving images. File system writing is disabled in this environment.
    """
    pass

@app.route('/')
def index():
    # Looks for 'index.html' in the 'templates' directory
    return render_template('index.html')

@app.route('/process_sketch', methods=['POST'])
def process_sketch():
    global c
    c += 1
    print(f"Request count: {c}")
    
    try:
        # Parse the incoming request
        data = request.get_json()
        image_data = data['image'] # Expecting the base64 data URL string

        # Preprocess the sketch
        sketch = preprocess_image(image_data)
        
        # Generate colored image
        colored_image = model.predict(sketch)

        # Postprocess the generated image
        colored_image_base64 = postprocess_image(colored_image)

        # Return the result
        return jsonify({'coloredImage': colored_image_base64})
    except Exception as e:
        # Log the error and return a 500 status
        error_message = f"Error processing sketch: {e}"
        print(error_message)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    # Mandatory for deployment: use the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
