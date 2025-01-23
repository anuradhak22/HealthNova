from flask import Flask, render_template, request, jsonify
import base64
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os

app = Flask(__name__)

# IBM WatsonX API Credentials
WATSONX_EU_APIKEY = "QMPOBePuaB7lDpB9Kj2bYLPcSmrV7GJi6cFi-EjGpMAi"
WATSONX_EU_PROJECT_ID = "07ecf28e-2ad4-4c82-8133-f2485ed03b0c"
URL = "https://us-south.ml.cloud.ibm.com"

credentials = Credentials(
    url=URL,
    api_key=WATSONX_EU_APIKEY
)

# Model initialization
model = ModelInference(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=credentials,
    project_id=WATSONX_EU_PROJECT_ID,
    params={"max_tokens": 50}
)

# Function to process the uploaded image
def process_uploaded_image(uploaded_file):
    content = uploaded_file.read()
    encoded_image = base64.b64encode(content).decode("utf-8")
    return encoded_image

# Function to generate the request body for the model
def augment_api_request_body(user_query, image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'You are a helpful assistant. Answer the following query about the product in the image: ' + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                }
            ]
        }
    ]
    return messages

# Routes
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            encoded_image = process_uploaded_image(file)

            # Query 1: What is the product and basic details?
            user_query = (
                "Identify the product in the image. Provide its name, brand, "
                "whether it is for consumption, application, or inhalation, "
                "recommended usage amount, and if it is harmful. Also, mention any "
                "body part it affects and its key ingredients."
            )
            messages = augment_api_request_body(user_query, encoded_image)
            response = model.chat(messages=messages)

            product_info = response['choices'][0]['message']['content']

            # Additional query for user-specific concerns (optional)
            user_query = "Are there any potential side effects or risks associated with this product?"
            messages = augment_api_request_body(user_query, encoded_image)
            response = model.chat(messages=messages)

            risks_info = response['choices'][0]['message']['content']

            return jsonify({
                'product_info': product_info,
                'risks_info': risks_info
            })

        except Exception as e:
            print(f"Error in API request: {e}")
            return jsonify({'error': 'Error processing request. Please try again later.'})
    else:
        return jsonify({'error': 'Invalid file format. Only .jpg, .jpeg, .png allowed.'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
