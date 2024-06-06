from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the TensorFlow saved model
model = tf.saved_model.load("gs://ml-model-storage-bucket/model-testing/")
infer = model.signatures["serving_default"]

# Define class names
class_names = [
   'aqua', 'coca cola', 'fanta', 'fruit tea', 'golda', 'mizone', 'nutri boost',
   'pocari sweet', 'pulpy', 's tee', 'sprite', 'tebs', 'teh botol', 'teh pucuk'
]

# Define the data augmentation and preprocessing layers as used in training
data_augmentation = tf.keras.Sequential([
   tf.keras.layers.RandomFlip('horizontal'),
   tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

def preprocess_image(image):
   img = Image.open(io.BytesIO(image))
   img = img.resize((224, 224))  # IMG_SIZE as defined during training
   img = np.array(img)   # Convert PIL image to NumPy array
   img = tf.expand_dims(img, axis=0)  # Add batch dimension
   img = data_augmentation(img)  # Apply data augmentation
   img = preprocess_input(img)  # Preprocess the augmented image
   return img

@app.route('/predict', methods=['POST'])
def predict():
   if 'file' not in request.files:
       return "No file part", 400
   file = request.files['file']
   if file.filename == '':
       return "No selected file", 400
   image = file.read()
   input_image = preprocess_image(image)
   predictions = infer(tf.constant(input_image))
   output_tensor_name = list(predictions.keys())[0]
   output_tensor = predictions[output_tensor_name].numpy()
   
   # Map predictions to class names
   predicted_class_indices = np.argmax(output_tensor, axis=1)
   predicted_classes = [class_names[idx] for idx in predicted_class_indices]
   confidence_scores = np.max(output_tensor, axis=1)

   # Return JSON response
   response = [
       {'class': predicted_classes[i]}
       for i in range(len(predicted_classes))
   ]
   return jsonify(response)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
