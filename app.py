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
    # Create a placeholder function if model fails to load to prevent startup crash
    def placeholder_predict(self, sketch): # <-- ADD 'self' HERE
        print("Using placeholder prediction (Model failed to load).")
        # Return a simple grayscale image array to match expected output shape
        return np.zeros((1, 256, 256, 3), dtype=np.float32) 
    model = type('DummyModel', (object,), {'predict': placeholder_predict})()


def preprocess_image(image_data):
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("L")
    image = image.resize((256,256))

    img = np.array(image).astype(np.float32)

    # Invert: black background, white lines
    img = 255 - img

    # Normalize to [-1,1]
    img = (img / 127.5) - 1.0

    return img[np.newaxis,...,np.newaxis]


def postprocess_image(model_output):
    image = model_output[0]

    # handle [0,1] or [-1,1] automatically
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = ((image + 1) * 127.5).astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
