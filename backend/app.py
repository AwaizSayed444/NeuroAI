from flask import Flask, request, jsonify
from chatbot_utils import predict_mood, fetch_questions, log_response

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    mood, confidence = predict_mood(user_input)
    questions = fetch_questions(mood)

    response_data = {
        "mood": mood,
        "confidence": confidence,
        "questions": questions
    }

    log_response({
        "user_input": user_input,
        "predicted_mood": mood,
        "confidence": confidence,
        "questions": questions
    })

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
