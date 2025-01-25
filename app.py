from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import base64
import fitz  # PyMuPDF
from io import BytesIO
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config["UPLOAD_FOLDER"] = "uploads/"

# IBM WatsonX API Credentials
WATSONX_EU_APIKEY = "RR2tQfVLLwcXESHCWjGc1wtAXdVbhVSIa6Y3rSNR-6Xz"
WATSONX_EU_PROJECT_ID = "c7864ee4-e791-4cd5-8a85-9d89bc94986b"
URL = "https://us-south.ml.cloud.ibm.com"

credentials = Credentials(url=URL, api_key=WATSONX_EU_APIKEY)
model = ModelInference(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=credentials,
    project_id=WATSONX_EU_PROJECT_ID,
    params={"max_tokens": 100}
)

# Predefined remedies for health issues
REMEDIES = {
    "fever": "Drink plenty of fluids and rest. You can also use a cold compress to reduce body temperature.",
    "cold": "Inhale steam with eucalyptus oil, drink warm fluids, and rest well.",
    "cough": "Drink warm honey lemon tea and keep yourself hydrated.",
    "headache": "Apply a cold pack or massage with peppermint oil. Rest in a dark, quiet room.",
    "vomiting": "Drink ginger tea or suck on ginger candy. Stay hydrated with small sips of water.",
    "backpain": "Apply a hot or cold compress. Practice light stretching or yoga.",
    "eyepain": "Rest your eyes, use artificial tears, and avoid screens. A cold compress may help.",
    "hi": "Hello, How can I help you?",
    "hello": "Hello, can I help you?",
    "how are you": "I am fine, how can I help you?"
}

# Helper functions
def encode_image(file):
    with open(file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(pdf_file)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

def augment_api_request_body(user_query, text_or_image):
    if isinstance(text_or_image, str):  # Text from PDF
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'You are a medical expert assistant. Analyze the uploaded medical report and answer the following: ' + user_query
                    },
                    {
                        "type": "text",
                        "text": text_or_image  # Text extracted from the PDF
                    }
                ]
            }
        ]
    else:  # Image
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'You are a helpful assistant. Answer the following query about the product in the image: ' + user_query
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{text_or_image}"}
                    }
                ]
            }
        ]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "").lower()
    response = "I'm sorry, I don't have a remedy for that issue. Please consult a doctor."

    for issue, remedy in REMEDIES.items():
        if issue in query:
            response = remedy
            break

    return jsonify({"response": response})

@app.route("/productanalyzer")
def product_analyzer():
    return render_template("productanalyzer.html")

@app.route("/summarization")
def medical_summary():
    return render_template("summarization.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_extension = file.filename.split('.')[-1].lower()
    if file_extension in ['jpg', 'jpeg', 'png']:  # Image file
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        encoded_image = encode_image(file_path)

        # Query 1: Extract product details
        query_1 = (
            "Identify the product in the image. Provide its name, brand, "
            "whether it is for consumption, application, or inhalation, "
            "recommended usage amount, and if it is harmful. Also, mention any "
            "body part it affects and its key ingredients."
        )
        messages_1 = augment_api_request_body(query_1, encoded_image)
        response_1 = model.chat(messages=messages_1)
        product_info = response_1['choices'][0]['message']['content']

        # Query 2: Identify risks
        query_2 = "Are there any potential side effects or risks associated with this product?"
        messages_2 = augment_api_request_body(query_2, encoded_image)
        response_2 = model.chat(messages=messages_2)
        risks_info = response_2['choices'][0]['message']['content']

        return jsonify({
            "product_info": product_info,
            "risks_info": risks_info
        })

    elif file_extension == 'pdf':  # PDF file
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        extracted_text = extract_text_from_pdf(file_path)

        # Query 1: Summarize medical report
        query_1 = (
            "Summarize the uploaded medical report. Mention the current condition of the user, "
            "any diagnosis mentioned, and overall health status."
        )
        messages_1 = augment_api_request_body(query_1, extracted_text)
        response_1 = model.chat(messages=messages_1)
        summary = response_1['choices'][0]['message']['content']

        # Query 2: Predict future tests and risks
        query_2 = (
            "Based on the medical report, predict any diagnostic tests the user might need in the future. "
            "Identify potential disease risks and suggest appropriate precautions."
        )
        messages_2 = augment_api_request_body(query_2, extracted_text)
        response_2 = model.chat(messages=messages_2)
        predictions = response_2['choices'][0]['message']['content']

        return jsonify({
            "summary": summary,
            "predictions": predictions
        })

    else:
        return jsonify({"error": "Unsupported file type. Only PDF and image files are allowed."}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
