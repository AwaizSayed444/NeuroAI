import os
import json
import pickle
import numpy as np
from datetime import datetime
from uuid import uuid4
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from flask_cors import CORS

# --- Custom Attention Layer ---
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
CORS(app)

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

with open(r"E:\AI ML Models\mood_keywords_grouped_enhanced.json", "r", encoding="utf-8") as f:
    mood_keywords = json.load(f)

# --- Process question data ---
question_data = {}
for entry in raw_data:
    category = entry.get("category", "").strip().lower()
    if not category:
        continue
    all_qs = []
    for doc in entry.get("questions", []):
        all_qs.extend([q.strip() for q in doc.get("questions", []) if isinstance(q, str) and len(q.strip()) > 10])
    if all_qs:
        question_data[category] = all_qs

# --- In-memory session store ---
sessions = {}

# --- Logging ---
def log_entry(data):
    path = "chat_log.json"
    logs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

# --- Mood Detection ---
def detect_mood_keywords(text):
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(k in text for k in keywords):
            return mood
    return None

def predict_mood(text):
    mood = detect_mood_keywords(text)
    if mood:
        return mood, 1.0
    emb = sbert.encode([text])
    emb = np.expand_dims(emb, axis=1)
    pred = model.predict(emb, verbose=0)
    idx = np.argmax(pred)
    conf = float(np.max(pred))
    mood = label_encoder.inverse_transform([idx])[0]
    return mood, conf

# --- Fetch Questions ---
def fetch_questions(mood, count=3):
    key = mood.strip().lower()
    questions = question_data.get(key, [])
    if not questions:
        return []
    return list(np.random.choice(questions, size=min(count, len(questions)), replace=False))

# --- Predict and Start Chat Session ---
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    mood, confidence = predict_mood(user_input)
    questions = fetch_questions(mood)

    if not questions:
        return jsonify({
            "reply": f"I'm here for you. Let's talk more about feeling {mood.lower()}."
        }), 200

    session_id = str(uuid4())
    sessions[session_id] = {
        "mood": mood,
        "confidence": confidence,
        "questions": questions,
        "current_index": 0,
        "answers": [],
        "user_input": user_input,
    }

    first_question = questions[0]
    return jsonify({
        "session_id": session_id,
        "reply": f"It seems you're feeling {mood.lower()}. Let's explore further:\n1. {first_question}",
        "finished": False
    }), 200

@app.route("/next", methods=["POST"])
def next_question():
    data = request.get_json()
    print("Received data at /next:", data) 
    session_id = data.get("session_id")
    answer = data.get("answer", "").strip().lower()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session ID"}), 400
    if not answer:
        return jsonify({"error": "Answer is required"}), 400

    session = sessions[session_id]

    # Check if user wants to end the assessment
    if answer in ["exit", "quit", "stop", "end"]:
        return jsonify({
            "reply": "Assessment complete. Thank you!",
            "finished": True
        })

    # Save the answer
    session["answers"].append(answer)

    # Move to next question
    session["current_index"] += 1
    index = session["current_index"]
    questions = session["questions"]

    # If current questions are finished, try to load more
    if index >= len(questions):
        mood = session["mood"]
        # Filter out already asked questions
        asked = set(questions)
        all_questions = set(question_data.get(mood.lower(), []))
        remaining = list(all_questions - asked)

        if remaining:
            new_questions = list(np.random.choice(remaining, size=min(3, len(remaining)), replace=False))
            session["questions"].extend(new_questions)
            questions = session["questions"]

        else:
            return jsonify({
                "reply": "You've answered all available questions. Type 'exit' to end the session or share more thoughts.",
                "finished": False
            })

    # Return the next question
    next_q = questions[session["current_index"]]
    return jsonify({"reply": next_q, "finished": False})
# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True, port=8000)
