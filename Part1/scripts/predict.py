import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Charger modèle et classes
model = load_model('mon_model_cnn.h5')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    class_id = np.argmax(preds[0])
    class_name = inv_class_indices[class_id]
    confidence = preds[0][class_id]
    print(f"Image: {os.path.basename(img_path)} -> Classe: {class_name} (Confiance: {confidence:.2f})")

# # Exemple d’utilisation
predict_image('test-dataset/test-dataset/class_1/45DAA32B-E90B-4A41-8EEE-F31E5D8A43CB.png')
