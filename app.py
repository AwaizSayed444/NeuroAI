import os
import json
import pickle
import numpy as np
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# --- Custom Attention Layer (must match training) ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model, Encoder, SBERT ---
print("📦 Loading mood classification model...")
model = load_model(r"E:\AI ML Models\mood_classifier_model.keras", custom_objects={"AttentionLayer": AttentionLayer})
print("✅ Model loaded!")

print("📦 Loading SBERT...")
sbert = SentenceTransformer("paraphrase-MiniLM-L12-v2")
print("✅ SBERT loaded!")

print("📦 Loading LabelEncoder...")
with open(r"E:\AI ML Models\mood_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
print("✅ LabelEncoder loaded!")

print("📦 Loading question data...")
with open(r"E:\AI ML Models\cleaned_questions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

question_data = {}
for entry in raw_data:
    category = entry.get("category", "").strip().lower()
    all_qs = []
    for doc in entry.get("questions", []):
        all_qs.extend([q.strip() for q in doc.get("questions", []) if isinstance(q, str) and len(q.strip()) > 10])
    if all_qs:
        question_data[category] = all_qs

print(f"✅ Loaded questions for {len(question_data)} mood categories")

# --- Helper: Fetch Questions ---
def fetch_questions(mood, count=3):
    key = mood.strip().lower()
    questions = question_data.get(key, [])
    return list(np.random.choice(questions, size=min(count, len(questions)), replace=False)) if questions else []

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Mood Classifier API!"})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": model is not None,
        "sbert_loaded": sbert is not None,
        "label_encoder_loaded": label_encoder is not None,
        "loaded_question_categories": list(question_data.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('input', '').strip()

        if not user_input:
            return jsonify({'error': 'Input text is required'}), 400

        embedding = sbert.encode([user_input])
        embedding = np.expand_dims(embedding, axis=1)  # Shape (1, 1, 384)

        predictions = model.predict(embedding)
        predicted_idx = int(np.argmax(predictions))
        predicted_mood = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(np.max(predictions))

        questions = fetch_questions(predicted_mood)

        return jsonify({
            'input': user_input,
            'predicted_mood': predicted_mood,
            'confidence': confidence,
            'questions': questions,
            'probabilities': predictions.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)
