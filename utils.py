import cv2
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, size=(256, 256)):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize and normalize
    image = cv2.resize(image, size)
    image = image / 255.0
    image = image.astype(np.float32)
    
    # Convert to PIL image
    image = Image.fromarray((image * 255).astype(np.uint8))
    
    return image
