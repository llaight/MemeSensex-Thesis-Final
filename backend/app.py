from flask import Flask, request, jsonify
from test_mode import run_inference
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Endpoint to handle image upload and return processed text and image tensor shape
@app.route('/process_predict', methods=['POST'])
def process_predict():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # convert to bytes
    image = request.files['image']
    image_bytes = image.read()

    # result
    result = run_inference(image_bytes)

    return jsonify({
        "status": "success",
        "message": "Image processed successfully and saved into the database",
        "data": result
    }), 201


if __name__ == '__main__':
    app.run(debug=True, port=5001)
