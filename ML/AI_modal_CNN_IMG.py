# ==============================================
# SIH Project: AI-Powered Crop Health Monitoring
# ==============================================
import sys
import os

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1️⃣ Workspace Paths
# -------------------------------
project_path = "sih_project"
image_path = r"D:\SIH_PROJECT\images"
sensor_csv = r"D:\SIH_PROJECT\sensor_data.csv"

# -------------------------------
# 2️⃣ Image Preprocessing
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15
)

train_gen = datagen.flow_from_directory(
    image_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    image_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Image preprocessing done!")

# -------------------------------
# 3️⃣ Sensor Data Preprocessing
# -------------------------------
sensor_df = pd.read_csv(sensor_csv, parse_dates=['timestamp'])
sensor_columns = ['soil_moisture', 'temp', 'humidity', 'leaf_wetness']
sensor_df[sensor_columns] = sensor_df[sensor_columns] / sensor_df[sensor_columns].max()
avg_sensor_values = sensor_df[sensor_columns].mean().values
print("Average sensor values:", avg_sensor_values)

# -------------------------------
# 4️⃣ CNN Model Definition
# -------------------------------
model = Sequential([
    Input(shape=(128,128,3)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Optional: concatenate sensor data here for fusion
    # We'll add it later in the hackathon demo
    
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 5️⃣ Train the Model
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------------
# 6️⃣ Save the Model
# -------------------------------
model.save(os.path.join(project_path, "crop_health_cnn.h5"))
print("Model saved at:", os.path.join(project_path, "crop_health_cnn.h5"))

# -------------------------------
# 7️⃣ Demo Prediction Example
# -------------------------------
from tensorflow.keras.preprocessing import image

# Load one image for demo
demo_img_path = r"D:\SIH_PROJECT\images\healthy\Pepper__bell___healthy\0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275.JPG"  # replace with an actual file
img = image.load_img(demo_img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
status = "Healthy" if pred[0][0] < 0.5 else "Unhealthy"
print("Prediction for demo image:", status)
