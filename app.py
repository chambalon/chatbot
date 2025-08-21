from flask import Flask, request, jsonify 
from chatbot import predict_intent, get_response

app = Flask(__name__)
@app.route('/chatbot', methods=['POST'])

def chat():
  user_input = request.json["message"]
  intent = predict_intent(user_input)
  response = get_response(intent)
  return jsonify({'response': response})

if __name__ == "__main__ ":
  app.run(debug=True)