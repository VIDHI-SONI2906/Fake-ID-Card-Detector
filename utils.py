import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# =========================
# TESSERACT CONFIGURATION
# =========================
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


# =========================
# IMAGE PREPROCESSING (CNN)
# =========================
IMG_HEIGHT = 150
IMG_WIDTH = 150

def preprocess_image(file):
    """
    Preprocess image for CNN model prediction.

    Supports:
    - PIL Image
    - Streamlit uploaded file
    - File path (string)
    """

    # ✅ Handle all input types
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)

    # Ensure image is RGB
    img = img.convert("RGB")

    # Resize
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to array
    img = img_to_array(img)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# =========================
# CNN PREDICTION FUNCTION
# =========================
def predict_image(model, file):
    """
    Predict whether ID card is Real or Fake.

    Returns:
    - label (str)
    - confidence (float)
    """

    img = preprocess_image(file)
    prediction = float(model.predict(img)[0][0])

    if prediction > 0.5:
        return "Real ID Card", prediction
    else:
        return "Fake ID Card", 1 - prediction


# =========================
# OCR FUNCTIONS
# =========================
def load_image_for_ocr(file):
    """
    Convert uploaded file to OpenCV format for OCR.
    """

    if isinstance(file, Image.Image):
        img = np.array(file)
    else:
        img = Image.open(file)
        img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def extract_text(img):
    """
    Extract text from image using Tesseract OCR.
    """

    if img is None:
        return "No image provided."

    # Convert to grayscale.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        text = pytesseract.image_to_string(img_gray)
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        return "Tesseract not found. Check installation path."

    except Exception as e:
        return f"OCR Error: {e}"