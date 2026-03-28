import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# ====================================================================
# 1. Define File Paths and Model
# ====================================================================

# Path to the trained model
model_path = 'model.h5'

# Path to the image you want to test
test_image_path = test_image_path = r"C:\Users\vidhi\Pictures\Saved Pictures\real img\Student Id Card Design real.jpg"
# Define the image dimensions used for training
img_height, img_width = 150, 150

# ====================================================================
# 2. Load the Model
# ====================================================================

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ====================================================================
# 3. Pre-process the Test Image
# ====================================================================

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction.
    """
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at '{image_path}'.")
        return None

    # Load the image with the correct size
    img = load_img(image_path, target_size=(img_height, img_width))

    # Convert the image to a NumPy array
    img_array = img_to_array(img)

    # Add a batch dimension (required by the model)
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale the pixel values to a range of [0, 1]
    img_array /= 255.0
    
    return img_array

# ====================================================================
# 4. Make a Prediction
# ====================================================================

preprocessed_image = preprocess_image(test_image_path)

if preprocessed_image is not None:
    # Make the prediction
    prediction = model.predict(preprocessed_image)

    # The prediction is a single value between 0 and 1
    probability = prediction[0][0]

    # Interpret the prediction
    if probability > 0.5:
        print("\nPrediction: The ID card is likely REAL.")
        print(f"Confidence Score: {probability:.2f}")
    else:
        print("\nPrediction: The ID card is likely FAKE.")
        print(f"Confidence Score: {1 - probability:.2f}")