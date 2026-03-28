Fake ID Card Detector:-

Detects Real vs Fake ID Cards using a CNN model, OpenCV, and Streamlit.

🚀 Features

1. Upload ID card image on Streamlit frontend to get instant prediction
2. Train your own CNN model on custom dataset
3. Test model by uploading individual images (URL or local path via fileimg.py)
4. Image preprocessing with OpenCV
5. Lightweight Python project with reusable modules

📁 Project Structure
Fake-ID-Card-Detector/
│── app.py               # Streamlit frontend
│── train_cnn.py         # Train CNN model
│── test_cnn.py          # Test model accuracy
│── fileimg.py           # Predict single image by URL or path
│── utils.py             # Helper functions
│── cnn_model_dataset/   # Dataset folder
│── requirements.txt
│── model.h5             # (Not included, download separately)
│── README.md
│── venv/                # Ignored

📥 Download Trained Model

 model.h5 is not on GitHub due to size (>100MB)
 
Download from Google Drive and place it in project root:

-> Fake-ID-Card-Detector/model.h5

🔧 Installation

1. Create virtual environment

-> python -m venv venv

-> venv\Scripts\activate

2. Install dependencies:

-> pip install -r requirements.txt

▶️ Run Streamlit Frontend

-> streamlit run app.py
<img width="1904" height="930" alt="Screenshot 2025-12-12 164248" src="https://github.com/user-attachments/assets/f81d3cc0-45c5-4b1d-bd5e-1a9db2f01261" />
<img width="1870" height="949" alt="Screenshot 2025-12-12 164404" src="https://github.com/user-attachments/assets/40a3f07a-cd88-4eb5-b875-0c1419172d76" />
<img width="1909" height="927" alt="Screenshot 2025-12-12 164428" src="https://github.com/user-attachments/assets/92c67396-a75b-4048-8e32-43a823e104d6" />

1. Upload an ID card image
2. Get prediction: Real / Fake

🧠 Train & Test Backend

1. Using Custom Dataset
   Organize your dataset:
   cnn_model_dataset/
│── train/
│     ├── real/
│     └── fake/
│── val/
      ├── real/
      └── fake/

Train model:

-> python train_cnn.py

Test model:

-> python test_cnn.py

This generates model.h5 for predictions.

2. Predict Single Image via URL or Path

Open fileimg.py and provide the image:

In the code update the image path:-

-> image_path = "path_or_url_to_id_card"

-> predict_image(image_path)

After this again you can test your provided image...

Works for online images or local files

Useful for testing without retraining the model


👤 Author

Vidhi Soni – AI/ML & Python Computer Vision Enthusiast
