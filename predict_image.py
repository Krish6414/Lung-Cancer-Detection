import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Define class labels
HISTO_CLASSES = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']
CT_CLASSES = ['adenocarcinoma', 'benign', 'large cell carcinoma', 'malignant', 'normal', 'squamous cell carcinoma']

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def predict_image(img_path, img_type):
    if img_type == 'histopathology':
        model_path = 'models/histopathology/histopathology_cnn_best.h5'
        target_size = (64, 64)
        class_labels = HISTO_CLASSES
    elif img_type == 'ct':
        model_path = 'models/ct_scans/ct_vgg16_best.h5'
        target_size = (224, 224)
        class_labels = CT_CLASSES
    else:
        raise ValueError("Invalid image type. Use 'histopathology' or 'ct'.")

    # Load model
    model = load_model(model_path)

    # Preprocess image
    img_array = load_and_preprocess_image(img_path, target_size)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"\n✅ Prediction: {predicted_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict lung cancer from a given image")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--type', required=True, choices=['histopathology', 'ct'], help='Type of image')
    args = parser.parse_args()

    predict_image(args.image, args.type)
