from flask import Flask, render_template, request
from chatbot1 import chatbot_response  # Import chatbot function

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["user_input"]  # Get user input from form
    bot_reply = chatbot_response(user_message)  # Get chatbot response
    return bot_reply  # Return response to frontend

if __name__ == "__main__":
    app.run(debug=True)
