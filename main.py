import os
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from utils import load_and_preprocess_image

def recognize_handwritten_text(image_path):
    # Load the TrOCR processor and model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Prepare the image for the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate the text from the image
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path_to_image")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)
    
    recognized_text = recognize_handwritten_text(image_path)
    print("Recognized Text:\n", recognized_text)
