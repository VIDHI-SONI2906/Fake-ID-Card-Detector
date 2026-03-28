import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable (VERY IMPORTANT!)
# Replace the path below with the correct path on your computer.
# For Windows, it might be 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# For macOS with Homebrew, it might be '/usr/local/bin/tesseract'
if os.name == 'nt':  # Checks if the operating system is Windows
    # Update this path to where you installed Tesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # This is a common path for Linux/macOS
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def load_image_for_ocr(file):
    """
    Loads an image from a Streamlit uploaded file object for OCR processing.
    Returns the image in OpenCV format (BGR).
    """
    img = Image.open(file)
    img = np.array(img)
    # Convert from RGB (PIL) to BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def extract_text(img):
    """
    Extracts text from an OpenCV image.
    The image is first converted to grayscale for better OCR results.
    """
    if img is None:
        return "No image provided."
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to extract text
    try:
        text = pytesseract.image_to_string(img_gray)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "Tesseract executable not found. Please check the path in utils.py."
    except Exception as e:
        return f"An error occurred during OCR: {e}"