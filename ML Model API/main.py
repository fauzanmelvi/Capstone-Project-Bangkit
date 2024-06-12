from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from google.cloud import firestore

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    # Load the TensorFlow saved model
    model = tf.saved_model.load("gs://ml-model-storage-bucket/model-testing/")
    infer = model.signatures["serving_default"]
except Exception as e:
    logging.error("Error loading TensorFlow model: %s", e)
    raise

# Define class names
class_names = [
   'aqua', 'coca cola', 'fanta', 'fruit tea', 'golda', 'mizone', 'nutri boost',
   'pocari sweet', 'pulpy', 's tee', 'sprite', 'tebs', 'teh botol', 'teh pucuk'
]

try:
    # Initialize Firestore client
    db = firestore.Client()
except Exception as e:
    logging.error("Error initializing Firestore client: %s", e)
    raise

# Define the data augmentation and preprocessing layers as used in training
data_augmentation = tf.keras.Sequential([
   tf.keras.layers.RandomFlip('horizontal'),
   tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

def preprocess_image(image):
    try:
        img = Image.open(io.BytesIO(image))
        img = img.resize((224, 224))  # IMG_SIZE as defined during training
        img = np.array(img)   # Convert PIL image to NumPy array
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
        img = data_augmentation(img)  # Apply data augmentation
        img = preprocess_input(img)  # Preprocess the augmented image
        return img
    except Exception as e:
        logging.error("Error preprocessing image: %s", e)
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    try:
        image = file.read()
        input_image = preprocess_image(image)
        predictions = infer(tf.constant(input_image))
        output_tensor_name = list(predictions.keys())[0]
        output_tensor = predictions[output_tensor_name].numpy()

        # Map predictions to class names
        predicted_class_indices = np.argmax(output_tensor, axis=1)
        predicted_class = class_names[predicted_class_indices[0]]
        confidence_score = np.max(output_tensor, axis=1)[0]

        # Fetch class details from Firestore
        class_details_ref = db.collection('classDetails').document(predicted_class)
        class_details = class_details_ref.get().to_dict()

        if class_details is None:
            return jsonify({'error': 'Class details not found'}), 404

        response = {
            'jenis minuman': predicted_class,
            'kandungan': class_details
        }
        return jsonify(response)
    except Exception as e:
        logging.error("Error processing prediction: %s", e)
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
