import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Layer

# Custom attention layer
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

# Define your model
def create_model():
    model = Sequential()
    model.add(Embedding(10000, 128))  # Example embedding layer
    model.add(LSTM(64, return_sequences=True))  # LSTM layer
    model.add(AttentionLayer())  # Custom Attention Layer
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 categories
    return model

# Train and save the model
def save_model():
    model = create_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Example: Training the model (you should replace this with your actual training data)
    # model.fit(X_train, y_train)  # Replace X_train and y_train with your actual training data

    # Save the model to a .keras file
    model.save('mood_classifier_model.keras')
    print("Model saved successfully!")

# Load the model (if needed)
def load_model_example():
    # Load the model with the custom layers defined earlier
    model = load_model('mood_classifier_model.keras', custom_objects={"AttentionLayer": AttentionLayer})

    # Test the loaded model (Example: Make a prediction)
    # predictions = model.predict(X_test)  # Replace X_test with actual test data
    print("Model loaded successfully!")

# Main code to save and load the model
if __name__ == "__main__":
    # Save the model
    save_model()

    # Load the model (optional)
    load_model_example()
