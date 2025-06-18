from flask import Flask, request, jsonify
from model import predict
import os
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from flask_cors import CORS
import logging
from download_model import download_model
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download model on startup
try:
    logger.info("Downloading model file...")
    download_model()
    logger.info("Model downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        logger.error(f"Failed to save image: {str(e)}")
        logger.error(traceback.format_exc())
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

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if model file exists
        model_path = "multi_condition_detector.pt"
        model_exists = os.path.exists(model_path)
        
        return jsonify({
            "status": "healthy",
            "model_available": model_exists,
            "model_path": model_path if model_exists else "not found"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    try:
        logger.info("Received prediction request")
        
        # Check if it's a JSON request with base64 image
        if request.is_json:
            logger.info("Processing JSON request with base64 image")
            data = request.get_json()
            if not data or 'image' not in data:
                logger.error("No image in JSON request")
                return jsonify({"error": "No image in request"}), 400
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image'])
                logger.info(f"Decoded base64 image, size: {len(image_data)} bytes")
                
                # Validate image can be opened
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(image_data))
                    logger.info(f"Image validated: {img.size} {img.mode}")
                except Exception as img_error:
                    logger.error(f"Invalid image format: {str(img_error)}")
                    return jsonify({"error": "Invalid image format"}), 400
                
                image_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
                with open(image_path, "wb") as f:
                    f.write(image_data)
                logger.info(f"Saved base64 image to {image_path}")
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {str(e)}")
                return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Check if it's a file upload request
        elif 'image' in request.files:
            logger.info("Processing file upload request")
            image_file = request.files['image']
            if not image_file.filename:
                logger.error("Empty filename")
                return jsonify({"error": "No filename provided"}), 400

            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            logger.info(f"Saved uploaded file to {image_path}")
            
            # Validate image can be opened
            try:
                from PIL import Image
                img = Image.open(image_path)
                logger.info(f"Image validated: {img.size} {img.mode}")
            except Exception as img_error:
                logger.error(f"Invalid image format: {str(img_error)}")
                # Clean up the invalid file
                if os.path.exists(image_path):
                    os.remove(image_path)
                return jsonify({"error": "Invalid image format"}), 400
        
        else:
            logger.error("Neither JSON nor file upload found")
            return jsonify({"error": "No image provided (neither file upload nor base64)"}), 400

        # Check if model file exists
        model_path = "multi_condition_detector.pt"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return jsonify({"error": "Model not available"}), 500

        try:
            logger.info("Starting prediction...")
            predictions = predict(image_path)
            logger.info(f"Generated predictions: {predictions}")
            return jsonify({"predictions": predictions})
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up file: {image_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

    except Exception as e:
        logger.error(f"Error in upload_and_predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/inpaint', methods=['POST'])
def inpaint_route():
    logger.info("📥 Received inpainting request")
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'boxes' not in data:
            logger.error("❌ Missing image or boxes in request")
            return jsonify({"error": "Missing image or boxes in request"}), 400

        logger.info("📤 Processing image and boxes...")
        image_path = save_base64_image(data['image'])
        logger.info(f"📤 Saved image to: {image_path}")
        
        boxes = data['boxes']
        method = data.get('method', 'telea')  # Default to telea if not specified
        logger.info(f"📤 Number of boxes to process: {len(boxes)}")
        logger.info(f"📤 Using inpainting method: {method}")

        try:
            logger.info("🎨 Starting inpainting process...")
            output_image = inpaint_image(image_path, boxes, method)
            logger.info("✅ Inpainting completed")

            logger.info("📤 Converting image to base64...")
            buffered = BytesIO()
            output_image.save(buffered, format="JPEG", quality=95)
            encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info("✅ Image converted to base64")
            
            return jsonify({
                "inpainted_image": encoded_img,
                "status": "success",
                "method": method
            })
        finally:
            # Clean up the uploaded file
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up file: {image_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error during inpainting: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Inpainting failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Use production settings
    app.run(host="0.0.0.0", port=port, debug=False)
 
