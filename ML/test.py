# ===============================
# Demo: Predict Healthy / Unhealthy
# ===============================
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Path to trained model
model_path = "sih_project/crop_health_cnn.h5"
model = load_model(model_path)

# Path to the image you want to test
demo_img_path = [ 
    r"D:\SIH_PROJECT\roma-tomato-leaves-unhealthy-on-bottom-v0-dfp6kbq68u1b1.webp",
    r"D:\SIH_PROJECT\images\healthy\Potato___healthy\2ccb9ee9-faac-4d32-9af5-29497fa2e028___RS_HL 1837.JPG"
]

for img_path in demo_img_path:
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

# Load and preprocess image
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

# Predict
    pred = model.predict(img_array)
    status = "Healthy" if pred[0][0] < 0.5 else "Unhealthy"

    print(f"Prediction for {os.path.basename(img_path)}: {status}")

