from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from google.cloud import firestore

app = Flask(__name__)

@app.route('/healthz')
def health_check():
    return 'OK', 200

logging.basicConfig(level=logging.INFO)

try:
    model = tf.saved_model.load("gs://ml-model-storage-bucket/modelfix/")
    infer = model.signatures["serving_default"]
except Exception as e:
    logging.error("Error loading TensorFlow model: %s", e)
    raise

class_names = [
   'Adem Sari', 'aqua', 'bear brand', 'cimory', 'coca cola', 'fanta', 'frestea', 'fruit tea', 'golda', 'good day', 'gunung', 'ichitan green tea', 'le minerale', 'milo', 'pocari', 'pulpy', 's tee', 'sprite', 'teh botol', 'teh gelas', 'teh kotak', 'teh pucuk', 'ultra milk moka'
]

try:
    db = firestore.Client()
except Exception as e:
    logging.error("Error initializing Firestore client: %s", e)
    raise

data_augmentation = tf.keras.Sequential([
   tf.keras.layers.RandomFlip('horizontal'),
   tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

def preprocess_image(image):
    try:
        img = Image.open(io.BytesIO(image))
        img = img.resize((224, 224))
        img = np.array(img)
        img = tf.expand_dims(img, axis=0)
        img = data_augmentation(img)
        img = preprocess_input(img)
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

        predicted_class_indices = np.argmax(output_tensor, axis=1)
        predicted_class = class_names[predicted_class_indices[0]]
        confidence_score = np.max(output_tensor, axis=1)[0]

        class_details_ref = db.collection('classDetails').document(predicted_class)
        class_details = class_details_ref.get().to_dict()

        if class_details is None:
            return jsonify({'error': 'Class details not found'}), 404
        
        gula_content = class_details.get('gula', 'N/A')
        warning_health = class_details.get('peringatan', 'N/A')

        response = {
            'jenis minuman': predicted_class,
            'kandungan gula': gula_content,
            'peringatan kesehatan': warning_health
        }
        return jsonify(response)
    except Exception as e:
        logging.error("Error processing prediction: %s", e)
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
