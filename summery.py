from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import fitz  # PyMuPDF
from io import BytesIO
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"

# IBM WatsonX API Credentials
WATSONX_EU_APIKEY = "itAdDn_d6l1oKJkEEuGNF7FA7U-rmZ5CjnBsIh5q4PHs"
WATSONX_EU_PROJECT_ID = "e23af05b-29f2-4fe9-aa47-0546c5b6bedb"
URL = "https://us-south.ml.cloud.ibm.com"

credentials = Credentials(url=URL, api_key=WATSONX_EU_APIKEY)
model = ModelInference(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=credentials,
    project_id=WATSONX_EU_PROJECT_ID,
    params={"max_tokens": 2}
)

# Helper function: Encode image to base64
def encode_image(file):
    with open(file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Helper function: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(pdf_file)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

# Helper function: Generate API request body
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
                        "text": 'You are a medical expert assistant. Analyze the uploaded medical report and answer the following: ' + user_query
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

@app.route("/upload_medical", methods=["POST"])
def upload_medical():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Check file type (image or PDF)
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:  # Image file
        # Save and process image file
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        encoded_image = encode_image(file_path)
# Query 1: Extract patient name, concise report summary, and current condition
        query_1 = (
"extract name of the patient only"
)
        messages_1 = augment_api_request_body(query_1, encoded_image)
        response_1 = model.chat(messages=messages_1)
        summary = response_1['choices'][0]['message']['content']

# Query 2: Predict future tests or risks
        query_2 = (
            "Based on the uploaded medical report, identify any diagnostic tests the user might need in the future. "
            "Additionally, predict potential disease risks and recommend appropriate precautions or preventive measures."
        )
        messages_2 = augment_api_request_body(query_2, encoded_image)
        response_2 = model.chat(messages=messages_2)
        predictions = response_2['choices'][0]['message']['content']

        

        

    elif file_extension in ['pdf']:  # PDF file
        # Save and process PDF file
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        extracted_text = extract_text_from_pdf(file_path)

        # Query 1: Summarize medical report (for text extracted from PDF)
        query_1 = (
            "Summarize the uploaded medical report. Mention the current condition of the user, "
            "any diagnosis mentioned, and overall health status."
        )
        messages_1 = augment_api_request_body(query_1, extracted_text)
        response_1 = model.chat(messages=messages_1)
        summary = response_1['choices'][0]['message']['content']

        # Query 2: Predict future tests and risks (for text extracted from PDF)
        query_2 = (
            "Based on the medical report, predict any diagnostic tests the user might need in the future. "
            "Identify potential disease risks and suggest appropriate precautions."
        )
        messages_2 = augment_api_request_body(query_2, extracted_text)
        response_2 = model.chat(messages=messages_2)
        predictions = response_2['choices'][0]['message']['content']

    else:
        return jsonify({"error": "Unsupported file type. Only PDF and image files are allowed."}), 400

    # Convert summary and predictions to formatted HTML
    formatted_summary = f"""
    <ul>
        <li><strong>Key Condition:</strong> {summary}</li>
    </ul>
    """

    formatted_predictions = "<ul>"
    for line in predictions.split("\n"):
        if line.strip():  # Skip empty lines
            if "test" in line.lower() or "risk" in line.lower():
                formatted_predictions += f"<li><strong>{line.strip()}</strong></li>"
            else:
                formatted_predictions += f"<li>{line.strip()}</li>"
    formatted_predictions += "</ul>"

    return jsonify({
        "summary": formatted_summary,
        "predictions": formatted_predictions
    })

if __name__ == "__main__":
    app.run(debug=True)
