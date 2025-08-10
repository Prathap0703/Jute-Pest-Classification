import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from class_labels import class_labels
import argparse

# Load the model once
model = tf.keras.models.load_model('xception_model_jute.keras')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # adjust size for Xception
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize between 0 and 1
    return img_array

def predict(img_path):
    processed_image = preprocess_image(img_path)
    preds = model.predict(processed_image)
    pred_index = np.argmax(preds)
    confidence = preds[0][pred_index]
    print(f"Predicted Class: {class_labels[pred_index]}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Jute Pest Class from Image')
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()
    
    predict(args.image)
