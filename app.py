from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Define models for different crops
crop_models = {
    "rice": {
        "CNN": tf.keras.models.load_model(r"D:\websitemaker - Copy\rice_models\cnn_rice.h5"),
        "EfficientNet": tf.keras.models.load_model(r"D:\websitemaker - Copy\rice_models\efficientnet_rice.h5"),
        "Inceptionv3": tf.keras.models.load_model(r"D:\websitemaker - Copy\rice_models\inceptionv3_rice.h5"),
        "VGG16": tf.keras.models.load_model(r"D:\websitemaker - Copy\rice_models\vgg16_rice.h5"),
    },
    "wheat": {
        "CNN": tf.keras.models.load_model(r"D:\websitemaker - Copy\wheat_models\cnn_wheat.h5"),
        "EfficientNet": tf.keras.models.load_model(r"D:\websitemaker - Copy\wheat_models\efficientnet_wheat.h5"),
        "Inceptionv3": tf.keras.models.load_model(r"D:\websitemaker - Copy\wheat_models\inceptionv3_wheat.h5"),
        "VGG16": tf.keras.models.load_model(r"D:\websitemaker - Copy\wheat_models\vgg16_wheat.h5"),
    },
    "corn": {
        "CNN": tf.keras.models.load_model(r"D:\websitemaker - Copy\corn_models\cnn_corn.h5"),
        "EfficientNet": tf.keras.models.load_model(r"D:\websitemaker - Copy\corn_models\efficientnet_corn.h5"),
        "Inceptionv3": tf.keras.models.load_model(r"D:\websitemaker - Copy\corn_models\inceptionv3_corn.h5"),
        "VGG16": tf.keras.models.load_model(r"D:\websitemaker - Copy\corn_models\vgg16_corn.h5"),
    },
}

# Load label encoders for different crops
label_encoders = {
    "rice": [line.strip() for line in open(r"D:\websitemaker\labels\label_encoder_rice.txt").read().split(',')],
    "wheat": [line.strip() for line in open(r"D:\websitemaker\labels\label_encoder_wheat.txt").read().split(',')],
    "corn": [line.strip() for line in open(r"D:\websitemaker\labels\label_encoder_corn.txt").read().split(',')],
}

UPLOAD_FOLDER = r"D:\websitemaker\upload_folder"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(imgpath, models, label_encoder):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    predictions = {}
    for model_name, model in models.items():
        preds = model.predict(img)
        predictions[model_name] = label_encoder[np.argmax(preds)]
    return predictions

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the crop type is selected
        crop_type = request.form.get("crop")
        if not crop_type or crop_type not in crop_models:
            return "Invalid crop type selected"
        
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load models and label encoder based on crop type
            selected_models = crop_models[crop_type]
            selected_label_encoder = label_encoders[crop_type]
            
            # Get predictions
            results = model_predict(filepath, selected_models, selected_label_encoder)
            return render_template("result.html", predictions=results, crop=crop_type)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
