from flask import Flask, request, jsonify, render_template
import base64, io, os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = None

tf.get_logger().setLevel("ERROR")
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def get_model():
    global model
    if model is None:
        print("🔁 Loading model...")
        model = tf.keras.models.load_model("Models/generator5.h5", compile=False)
        print("✅ Model loaded.")
    return model

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("RGBA")
    bg = Image.new("RGBA", image.size, (255,255,255,255))
    image = Image.alpha_composite(bg, image).convert("L")
    image = image.resize((256,256))
    img = np.array(image)/255.0
    return img[np.newaxis,...,np.newaxis]

def postprocess_image(model_output):
    image = model_output[0]
    image = (image*255).astype(np.uint8) if image.max()<=1 else ((image+1)*127.5).astype(np.uint8)
    if image.shape[-1]==1:
        image = np.repeat(image,3,axis=-1)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_sketch", methods=["POST"])
def process_sketch():
    try:
        data = request.get_json()
        sketch = preprocess_image(data["image"])
        model_instance = get_model()
        output = model_instance.predict(sketch)
        encoded = postprocess_image(output)
        return jsonify({"coloredImage": encoded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
