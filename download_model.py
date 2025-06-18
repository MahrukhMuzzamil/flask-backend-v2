import os
import requests
from tqdm import tqdm
import gdown
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    # Google Drive file ID - you'll need to replace this with your actual file ID
    MODEL_FILE_ID = "1WPu5NMKlyVMx5uvs8GYKGECoiqTJ90Mm"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "multi_condition_detector.pt")

    if os.path.exists(MODEL_PATH):
        logger.info("Model file already exists.")
        return

    logger.info("Downloading model file from Google Drive...")
    try:
        # Download using gdown
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully!")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 
