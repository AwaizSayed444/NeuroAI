from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define your custom AttentionLayer (must match training!)
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

# Initialize Flask app
app = Flask(__name__)

# Load model with custom layer
model = load_model(r'E:\AI ML Models\moods_classifier_model.keras', custom_objects={'AttentionLayer': AttentionLayer})

# Load tokenizer or any necessary preprocessing tools
with open('mood_encoder.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Model loaded successfully!")

# Preprocess input to match model input format (tokenization and padding)
def preprocess_input(text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences to match the model's input shape
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per your model's requirement
    return padded_sequences

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Mood Classifier API!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        user_input = data['input']  # Expecting input as a string

        # Preprocess the input
        processed_input = preprocess_input(user_input)

        # Make prediction
        predictions = model.predict(processed_input)
        predicted_class = int(np.argmax(predictions))  # Get the class with the highest probability

        # Mood labels (make sure they match the class indices from your model)
        mood_labels = ["Depression & Anxiety", "Personality & Behaviour", "Stress & Coping"]

        # Return the prediction and probabilities
        return jsonify({
            'prediction': mood_labels[predicted_class],
            'probabilities': predictions.tolist()  # Return class probabilities as a list
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
