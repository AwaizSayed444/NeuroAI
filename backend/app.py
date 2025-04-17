from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np

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
model = load_model(r'E:\MiniProjectAi\NeuroAI\backend\model\mood_classifier_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
print("Model loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Mood Classifier API!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['input'])  # Expecting input as a list of integers

        # Example padding if needed (should match what was used during training)
        # from tensorflow.keras.preprocessing.sequence import pad_sequences
        # input_data = pad_sequences([input_data], maxlen=MAX_LEN)

        input_data = np.expand_dims(input_data, axis=0)
        predictions = model.predict(input_data)
        predicted_class = int(np.argmax(predictions))

        return jsonify({
            'prediction': predicted_class,
            'probabilities': predictions.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
