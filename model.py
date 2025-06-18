# import torch
# from torchvision import transforms
# from PIL import Image
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision import models

# LABEL_MAP = {
#     1: "Frackels",
#     2: "Acne scars",
#     3: "Mole and Tags",
#     4: "Acne"
# }

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def load_model(model_path, num_classes):
#     model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model

# model = load_model("multi_condition_detector.pt", num_classes=len(LABEL_MAP) + 1)

# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0)
    
#     # Get image dimensions for normalization
#     img_width, img_height = image.size

#     with torch.no_grad():
#         predictions = model(image_tensor)[0]

#     results = []
#     for i, score in enumerate(predictions["scores"]):
#         if score >= 0.1:  # Confidence threshold
#             label_idx = predictions["labels"][i].item()
#             label_name = LABEL_MAP.get(label_idx, "Unknown")
#             box = predictions["boxes"][i].tolist()
#             # Normalize box coordinates
#             normalized_box = [
#                 box[0] / img_width,  # x1
#                 box[1] / img_height, # y1
#                 box[2] / img_width,  # x2
#                 box[3] / img_height  # y2
#             ]
#             results.append({
#                 "label": label_name,
#                 "score": round(score.item(), 2),
#                 "box": normalized_box
#             })

#     return results

import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
import gc
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    1: "Frackels",
    2: "Acne scars",
    3: "Mole and Tags",
    4: "Acne"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variable to store the model
_model = None

def load_model(model_path, num_classes):
    global _model
    try:
        if _model is None:
            logger.info("Loading model from: %s", model_path)
            
            # Verify model file exists
            if not os.path.exists(model_path):
                logger.error("Model file not found at: %s", model_path)
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Check file size
            file_size = os.path.getsize(model_path)
            logger.info("Model file size: %d bytes", file_size)
            
            if file_size == 0:
                raise ValueError("Model file is empty")
            
            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Load model with CPU
            logger.info("Creating model architecture...")
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # Load state dict with CPU
            logger.info("Loading model state dict...")
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            
            # Store model in global variable
            _model = model
            
            # Clear memory
            del state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model loaded successfully")
        
        return _model
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        logger.error(traceback.format_exc())
        raise

def predict(image_path):
    try:
        global _model
        if _model is None:
            logger.info("Model not loaded, loading now...")
            _model = load_model("multi_condition_detector.pt", num_classes=len(LABEL_MAP) + 1)
        
        logger.info("Processing image: %s", image_path)
        
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check image file size
        file_size = os.path.getsize(image_path)
        logger.info("Image file size: %d bytes", file_size)
        
        if file_size == 0:
            raise ValueError("Image file is empty")
        
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        logger.info("Image dimensions: %dx%d", img_width, img_height)

        # Resize for model input
        resized_image = image.resize((224, 224))
        image_tensor = transform(image).unsqueeze(0)
        logger.info("Image tensor shape: %s", image_tensor.shape)

        with torch.no_grad():
            logger.info("Running model inference...")
            predictions = _model(image_tensor)[0]
            logger.info("Model inference completed")

        results = []
        for i, score in enumerate(predictions["scores"]):
            label_idx = predictions["labels"][i].item()
            label_name = LABEL_MAP.get(label_idx, "Unknown")

            # Apply custom confidence thresholds
            if (label_name == "Mole and Tags" and score < 0.2) or (label_name != "Mole and Tags" and score < 0.1):
                continue

            box = predictions["boxes"][i].tolist()

            # Scale box coordinates back to original image dimensions
            scale_x = img_width / 224
            scale_y = img_height / 224
            scaled_box = [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y
            ]

            # Normalize to original image dimensions
            normalized_box = [
                scaled_box[0] / img_width,
                scaled_box[1] / img_height,
                scaled_box[2] / img_width,
                scaled_box[3] / img_height
            ]

            results.append({
                "label": label_name,
                "score": round(score.item(), 2),
                "box": normalized_box
            })

        # Clear memory
        del image_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Prediction completed with %d results", len(results))
        return results
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        logger.error(traceback.format_exc())
        raise



