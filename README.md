# 🌿 Jute Pest Classification

This project is an automated image classification system for identifying various jute pests using deep learning models.  
It aims to assist farmers and agricultural experts in early pest detection to reduce crop losses and improve yield.

---

## 🚀 Live Demo
The application is deployed and accessible here:  
**[Jute Pest Classifier on Streamlit Cloud](https://jute-pest-classification.streamlit.app/)**

---

## ✨ Features
- Classifies **17 different jute pest categories**.
- Built using a **fine-tuned Xception** deep learning model.
- Provides **confidence scores** for predictions.
- **User-friendly web interface** with Streamlit for easy image upload and prediction.
- Performance evaluated with **accuracy**, **precision**, **recall**, **confusion matrix**, and **AUC curves**.

---

## 🛠 Tech Stack
- **Python** 3.11
- **TensorFlow/Keras**
- **Streamlit**
- **gdown** for model downloading from Google Drive
- **Pillow** for image processing
- **NumPy** for numerical operations

---

## 📌 How It Works
1. Upload a jute pest image (JPG/PNG).
2. The model preprocesses the image to `299x299`.
3. Predictions are made using the fine-tuned Xception model.
4. Results display the predicted pest class and the confidence score.
5. A probability bar chart shows the likelihood for all pest classes.

---

## 👨‍💻 Developed By
Prathap V & Team  
_Jute Pest Classification Project_

---
