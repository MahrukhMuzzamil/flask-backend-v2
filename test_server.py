import requests
import base64
import json
from PIL import Image
import io

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get('https://flask-backend-v2.onrender.com/health')
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint with a simple image"""
    try:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Send request
        data = {'image': img_base64}
        response = requests.post(
            'https://flask-backend-v2.onrender.com/predict',
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"Prediction status: {response.status_code}")
        print(f"Prediction response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Flask server...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_health()
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    prediction_ok = test_prediction()
    
    print(f"\nResults:")
    print(f"Health check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Prediction: {'✅ PASS' if prediction_ok else '❌ FAIL'}") 