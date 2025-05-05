from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)

# Load models (make sure paths are correct relative to this file)
HISTO_MODEL_PATH = os.path.join("models", "histopathology", "histopathology_cnn_best.h5")
CT_MODEL_PATH = os.path.join("models", "ct_scans", "ct_vgg16_best.h5")

histo_model = load_model(HISTO_MODEL_PATH)
ct_model = load_model(CT_MODEL_PATH)

# Class labels
HISTO_CLASSES = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']
CT_CLASSES = ['Adenocarcinoma', 'Benign', 'Large Cell Carcinoma', 'Malignant', 'Normal', 'Squamous Cell Carcinoma']

def preprocess_image(file, target_size=(224, 224)):
    img = load_img(BytesIO(file.read()), target_size=target_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    histo_img = request.files.get('histopath')
    ct_img = request.files.get('ctscan')

    if histo_img and histo_img.filename != '':
        img = preprocess_image(histo_img)
        pred = np.argmax(histo_model.predict(img), axis=1)[0]
        prediction_class = HISTO_CLASSES[pred]
        image_type = "Histopathology"
    elif ct_img and ct_img.filename != '':
        img = preprocess_image(ct_img)
        pred = np.argmax(ct_model.predict(img), axis=1)[0]
        prediction_class = CT_CLASSES[pred]
        image_type = "CT Scan"
    else:
        prediction_class = "No image provided"
        image_type = "None"

    # Lung cancer status
    if prediction_class.lower() == 'normal' or prediction_class.lower() == 'benign':
        cancer_status = "No Lung Cancer Detected ✅"
    else:
        cancer_status = "Lung Cancer Detected ⚠️"

    return render_template('index.html',
                           name=name,
                           age=age,
                           gender=gender,
                           image_type=image_type,
                           prediction=prediction_class,
                           cancer_status=cancer_status)

if __name__ == '__main__':
    app.run(debug=True)
