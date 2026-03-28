# 🪪 Fake ID Card Detector

A Computer Vision project that detects whether an ID card is **Real or Fake** using a **Convolutional Neural Network (CNN)**, **OpenCV**, and a **Streamlit-based UI**.

---

## 📌 Overview

This project allows users to upload an ID card image and instantly receive a prediction (Real/Fake). It also provides functionality to train a custom CNN model and test it on new data.

The system is designed to demonstrate practical applications of Deep Learning in identity verification.

---

## 🚀 Features

* 📤 Upload ID card image via Streamlit UI
* 🤖 Predict whether the ID is **Real or Fake**
* 🧠 Train your own CNN model on a custom dataset
* 🧪 Test model performance on validation data
* 🌐 Predict images using URL or local file path
* ⚡ Fast and lightweight implementation using Python

---

## 🗂️ Project Structure

```
Fake-ID-Card-Detector/
│── app.py                # Streamlit frontend
│── train_cnn.py          # Train CNN model
│── test_cnn.py           # Evaluate model
│── fileimg.py            # Predict image via URL/path
│── utils.py              # Helper functions
│── cnn_model_dataset/    # Dataset (train + validation)
│── requirements.txt      # Dependencies
│── model.h5              # Trained model (download separately)
│── README.md
│── venv/                 # Virtual environment (ignored)
```

---

## 📥 Download Trained Model

The trained model (`model.h5`) is not included in this repository due to size limitations.

👉 Download it from Google Drive:
**https://drive.google.com/file/d/1gN2ypbv4tmt2DtL4zIPYnugwD_sRMMME/view?usp=sharing**

After downloading, place it in the project root directory:

```
Fake-ID-Card-Detector/model.h5
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone <your-repo-link>
cd Fake-ID-Card-Detector
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application (Frontend)

```
streamlit run app.py
```

### Steps:

1. Upload an ID card image
2. Click predict
3. Get output → **Real / Fake**

---

## 🧠 Model Training & Testing

### 📁 Dataset Structure

```
cnn_model_dataset/
│── train/
│   ├── real/
│   └── fake/
│── val/
    ├── real/
    └── fake/
```

---

### 🔹 Train Model

```
python train_cnn.py
```

---

### 🔹 Test Model

```
python test_cnn.py
```

This will evaluate model performance and generate predictions.

---

## 🖼️ Predict Single Image (Without UI)

You can test any image directly using:

```
fileimg.py
```

Update this line in the file:

```
image_path = "path_or_url_to_id_card"
```

Then run:

```
python fileimg.py
```

✔ Works with:

* Local file paths
* Image URLs

---

## 📸 Sample Output

* Upload ID card image
* Model predicts: **Real / Fake**

<img width="1919" height="1020" alt="Screenshot 2026-03-29 011845" src="https://github.com/user-attachments/assets/d554a547-b904-45d2-a0fc-e03581e5ad82" />

<img width="1892" height="639" alt="image" src="https://github.com/user-attachments/assets/f5dae1a1-1c19-4110-b58c-f7b774729fc8" />


---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* Streamlit
* NumPy

---

## 📌 Notes

* Dataset is included for training purposes
* Large files like model are provided separately
* Virtual environment (`venv`) is excluded from version control

---

## 👤 Author

**Vidhi Soni**
AI/ML & Computer Vision Enthusiast

---

## ⭐ Acknowledgment

This project is developed as part of academic submission and demonstrates the application of deep learning in real-world identity verification.

---
