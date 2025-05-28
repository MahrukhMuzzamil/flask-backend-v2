from flask import Flask, request, jsonify
from model import predict
import os
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_base64_image(base64_str, filename="input.jpg"):
    try:
        image_data = base64.b64decode(base64_str)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        raise ValueError(f"Failed to save image: {str(e)}")

def create_mask(img_shape, boxes, feather=5):
    """
    Create a smooth mask for inpainting with feathered edges
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    h, w = img_shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = [int(box[i] * [w, h, w, h][i]) for i in range(4)]
        # Add padding to ensure we don't get artifacts at the edges
        x1, y1 = max(0, x1 - feather), max(0, y1 - feather)
        x2, y2 = min(w, x2 + feather), min(h, y2 + feather)
        
        # Create a temporary mask for this box
        temp_mask = np.zeros_like(mask)
        temp_mask[y1:y2, x1:x2] = 255
        
        # Apply Gaussian blur to feather the edges
        temp_mask = cv2.GaussianBlur(temp_mask, (feather*2+1, feather*2+1), 0)
        
        # Combine with main mask
        mask = cv2.bitwise_or(mask, temp_mask)

    return mask

def inpaint_image(image_path, boxes, method='telea'):
    """
    Inpaint the image using the specified method
    methods: 'telea' (Telea's algorithm) or 'ns' (Navier-Stokes)
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")

        # Create mask
        mask = create_mask(img.shape, boxes)
        
        # Choose inpainting method
        if method.lower() == 'ns':
            inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        else:  # default to telea
            inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Convert to RGB for PIL
        rgb_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)
    
    except Exception as e:
        raise ValueError(f"Inpainting failed: {str(e)}")

@app.route('/')
def home():
    return "Flask API is running! Use POST /predict to send an image or POST /inpaint for inpainting."

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    predictions = predict(image_path)
    return jsonify({"predictions": predictions})

@app.route('/inpaint', methods=['POST'])
def inpaint_route():
    print("📥 Received inpainting request")
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'boxes' not in data:
            print("❌ Missing image or boxes in request")
            return jsonify({"error": "Missing image or boxes in request"}), 400

        print("📤 Processing image and boxes...")
        image_path = save_base64_image(data['image'])
        print(f"📤 Saved image to: {image_path}")
        
        boxes = data['boxes']
        method = data.get('method', 'telea')  # Default to telea if not specified
        print(f"📤 Number of boxes to process: {len(boxes)}")
        print(f"📤 Using inpainting method: {method}")

        print("🎨 Starting inpainting process...")
        output_image = inpaint_image(image_path, boxes, method)
        print("✅ Inpainting completed")

        print("📤 Converting image to base64...")
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("✅ Image converted to base64")
        
        return jsonify({
            "inpainted_image": encoded_img,
            "status": "success",
            "method": method
        })

    except Exception as e:
        print(f"❌ Error during inpainting: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
 
