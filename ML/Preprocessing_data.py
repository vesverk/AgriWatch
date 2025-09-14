import os 
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

project_path = "SIH_PROJECT"
image_path = r"D:\SIH_PROJECT\images"


datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    horizontal_flip = True,
    rotation_range = 15
)

train_gen = datagen.flow_from_directory(
    image_path,
    target_size = (128,128),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training',
    shuffle = True
)

val_gen = datagen.flow_from_directory(
    image_path,
    target_size= (128,128),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation',
    shuffle = True
)

print("image preprocessing done!")


#sensonr data proprosessing

sensor_file = r"D:\SIH_PROJECT\sensor_data.csv"

# Load CSV
sensor_df = pd.read_csv(sensor_file, parse_dates=['timestamp'])

# Correct column names
sensor_columns = ['soil_moisture', 'temp', 'humidity', 'leaf_wetness']

# Normalize sensor values to [0,1]
sensor_df[sensor_columns] = sensor_df[sensor_columns] / sensor_df[sensor_columns].max()

# Show first 5 rows
print(sensor_df.head())

# Compute average sensor values (for demo fusion)
avg_sensor_values = sensor_df[sensor_columns].mean().values
print("Average sensor values:", avg_sensor_values)
