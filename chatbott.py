from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Predefined remedies for health issues
REMEDIES = {
    "fever": "Drink plenty of fluids and rest. You can also use a cold compress to reduce body temperature.",
    "cold": "Inhale steam with eucalyptus oil, drink warm fluids, and rest well.",
    "cough": "Drink warm honey lemon tea and keep yourself hydrated.",
    "headache": "Apply a cold pack or massage with peppermint oil. Rest in a dark, quiet room.",
    "vomiting": "Drink ginger tea or suck on ginger candy. Stay hydrated with small sips of water.",
    "backpain": "Apply a hot or cold compress. Practice light stretching or yoga.",
    "eyepain": "Rest your eyes, use artificial tears, and avoid screens. A cold compress may help.",
    "hi":"How can i help you!",
    "hello":"How can i help you!",
    "how are you":"I am fine how can i help you?"
}

@app.route("/chat", methods=["POST"])
def chat():
    # Get the JSON payload from the frontend
    data = request.json
    query = data.get("query", "").lower()  # Get the user's query
    response = "I'm sorry, I don't have a remedy for that issue. Please consult a doctor."

    # Search for a matching remedy
    for issue, remedy in REMEDIES.items():
        if issue in query:
            response = remedy
            break

    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)  # Run on all available network interfaces
