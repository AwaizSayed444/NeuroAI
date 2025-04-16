import os
import re
import json
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# --- Custom Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# --- Load model & assets ---
try:
    model = load_model("moods_classifier_model.keras", custom_objects={"AttentionLayer": AttentionLayer})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

sbert = SentenceTransformer("paraphrase-MiniLM-L12-v2")

with open("mood_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("mood_keywords.json", "r", encoding="utf-8") as f:
    mood_keywords = json.load(f)

with open("cleaned_questions.json", "r", encoding="utf-8") as f:
    raw_question_data = json.load(f)

# --- Build question bank from cleaned_questions.json ---
question_data = {}
for entry in raw_question_data:
    category = entry.get("category", "").strip().lower()
    question_data[category] = []
    for doc in entry.get("questions", []):
        for q in doc.get("questions", []):
            if isinstance(q, str) and len(q.strip()) > 10:
                question_data[category].append(q.strip())

# --- Logging ---
def log_response(entry):
    LOG_FILE = "chat_log.json"
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

# --- Mood Detection ---
def detect_mood_keywords(text):
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(keyword in text for keyword in keywords):
            return mood
    return None

def predict_mood(text):
    mood = detect_mood_keywords(text)
    if mood:
        print(f"📌 Detected mood from keywords: {mood}")
        return mood, 1.0
    emb = sbert.encode([text])
    emb = np.expand_dims(emb, axis=1)
    pred = model.predict(emb, verbose=0)
    mood_idx = np.argmax(pred)
    confidence = float(np.max(pred))
    mood = label_encoder.inverse_transform([mood_idx])[0]
    print(f"🤖 Predicted mood: {mood} (confidence: {confidence:.2f})")
    return mood, confidence

# --- Fetch Questions ---
def fetch_questions(mood, count=3):
    mood_key = mood.strip().lower()
    available_categories = {k.lower(): k for k in question_data}
    matched_key = available_categories.get(mood_key)

    if not matched_key:
        print(f"⚠️ No questions found for mood '{mood}'.")
        return []

    all_qs = question_data[matched_key]
    return list(np.random.choice(all_qs, size=min(count, len(all_qs)), replace=False))

# --- Main Chatbot ---
def chatbot():
    print("🧠 MoodBot is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ["exit", "quit"]:
            print("👋 Take care. I'm here whenever you need support.")
            break

        mood, confidence = predict_mood(user_input)
        questions = fetch_questions(mood)

        if not questions:
            print(f"🧠 MoodBot: I'm here to support you. Would you like to share more about how you're feeling?\n")
            continue

        print(f"\n🧠 MoodBot ({mood.title()}): It sounds like you're feeling {mood.lower()}. Let's go through a few questions:\n")
        for i, question in enumerate(questions, 1):
            answer = input(f"{i}. {question}\nYou: ").strip()
            log_response({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "predicted_mood": mood,
                "confidence": confidence,
                "question": question,
                "answer": answer
            })

        print("\n✅ Thank you for sharing. You can continue chatting or type 'exit' to leave.\n")

# --- Run ---
if __name__ == "__main__":
    chatbot()
