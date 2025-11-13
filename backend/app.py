from flask import Flask, request, jsonify, send_from_directory, make_response
from test_mode import run_inference
from flask_cors import CORS
import os
import mimetypes
import traceback

# Initialize mimetypes
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

backend_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(backend_dir, "../frontend/memesensex-frontend/build")

app = Flask(__name__, static_folder=build_dir, static_url_path="")

# CRITICAL: More permissive CORS for tunnels
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# CRITICAL: Increase max content length for images through tunnels
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

@app.after_request
def after_request(response):
    """Add headers for tunnel compatibility"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    
    # Fix MIME types for tunnels
    if response.content_type == 'text/plain':
        if request.path.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript'
        elif request.path.endswith('.css'):
            response.headers['Content-Type'] = 'text/css'
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"}), 200

@app.route('/process_predict', methods=['POST', 'OPTIONS'])
def process_predict():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image = request.files['image']
        
        if image.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        image_bytes = image.read()
        result = run_inference(image_bytes)

        print(f"Processed result: {result}")

        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "data": {
                "prediction": result.get("prediction"),
                "raw_text": result.get("raw_text", ""),
                "clean_text": result.get("clean_text", ""),
                "probabilities": result.get("probabilities", []),
                "confidence": result.get("confidence", 0)
            }
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/static/<path:path>")
def serve_static(path):
    try:
        if path.endswith('.js'):
            mimetype = 'application/javascript'
        elif path.endswith('.css'):
            mimetype = 'text/css'
        else:
            mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        
        response = make_response(send_from_directory(
            os.path.join(app.static_folder, "static"), 
            path
        ))
        response.headers['Content-Type'] = mimetype
        return response
    except:
        return "Not found", 404

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    try:
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")
    except:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == '__main__':
    print(f"Static folder: {build_dir}")
    print(f"Starting server on 0.0.0.0:5001")
    
    # CRITICAL: Use threaded=True for tunnel compatibility
    app.run(debug=False, port=5001, host='0.0.0.0', threaded=True)