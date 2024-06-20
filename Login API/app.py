import os
import logging
from flask import Flask, request, jsonify, session
from google.cloud import firestore
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

try:
    db = firestore.Client()
except Exception as e:
    logging.error("Error initializing Firestore client: %s", e)
    raise

app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')

class UserStore:
    def add_user(self, name, email, password):
        user_ref = db.collection('user-login').document(email)
        user_ref.set({'name': name, 'email': email, 'password': password})

    def get_user(self, email):
        user_ref = db.collection('user-login').document(email)
        user_doc = user_ref.get()
        if user_doc.exists:
            return user_doc.to_dict()
        else:
            return None

user_store = UserStore()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({'error': 'Name, email, and password are required!'}), 400
    
    if user_store.get_user(email):
        return jsonify({'error': 'User already exists!'}), 400
    
    hashed_password = generate_password_hash(password)
    user_store.add_user(name, email, hashed_password)
    
    return jsonify({'message': 'User registered successfully!'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required!'}), 400
    
    user = user_store.get_user(email)
    if user and check_password_hash(user['password'], password):
        session['user'] = email
        return jsonify({'message': 'Logged in successfully!'}), 200
    
    return jsonify({'error': 'Invalid credentials!'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'message': 'Logged out successfully!'}), 200

@app.route('/status', methods=['GET'])
def status():
    if 'user' in session:
        return jsonify({'message': f'Logged in as {session["user"]}'}), 200
    return jsonify({'message': 'Not logged in!'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
