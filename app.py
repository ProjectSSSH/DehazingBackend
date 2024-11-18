from flask import Flask, request, jsonify, send_file
from io import BytesIO
from model import model  # Ensure model is loaded in model.py
from utils import load_image, image_to_tensor, tensor_to_image, dark_channel_prior, atmospheric_light, transmission_estimate
import torch
from PIL import Image
import numpy as np

app = Flask(__name__)

# Ensure model is set to evaluation mode for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.route('/dehaze', methods=['POST'])
def dehaze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']

    try:
        # Load and preprocess the image
        image = load_image(file)
        hazy_tensor = image_to_tensor(image).to(device)

        # Calculate dark channel prior and other required features
        dark_channel = dark_channel_prior(image)
        A = atmospheric_light(image, dark_channel)
        transmission = transmission_estimate(image, A)
        dcp_tensor = image_to_tensor(transmission).to(device)

        # Run inference using the model
        with torch.no_grad():
            dehazed_tensor = model(hazy_tensor, dcp_tensor)

        # Convert output tensor to image
        dehazed_image = tensor_to_image(dehazed_tensor)
        dehazed_image = (dehazed_image * 255).astype(np.uint8)  # Convert to 8-bit format for PIL compatibility


        # Prepare the image for response
        img_io = BytesIO()
        Image.fromarray(dehazed_image).save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
