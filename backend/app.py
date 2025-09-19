import base64
from flask import Flask, request, jsonify
from test_mode import run_inference
from flask_cors import CORS
import os
import sqlite3
import base64

app = Flask(__name__)
CORS(app)

# Initialize database
def init_db():
    conn= sqlite3.connect('memesensex_db')
    cursor= conn.cursor()
    return conn, cursor

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

    #save to db
    conn, cursor = init_db()
    cursor.execute("""
        INSERT INTO meme_tb(image, probabilities, prediction, raw_text, clean_text)
        VALUES(?, ?, ?, ?, ?)
    """, (image_bytes, str(result['probabilities']), result['prediction'], result['raw_text'], result['clean_text']))
    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "message": "Image processed successfully and saved into the database",
        "data": result
    }), 201

# Endpoint to retrieve logged data
@app.route('/log_data', methods=['GET'])
def log_data():
    conn, cursor= init_db()
    cursor.execute("SELECT * FROM meme_tb")
    rows= cursor.fetchall()
    conn.close()

    data= []
    for row in rows:
        data.append({
            'id': row[0],
            'image': base64.b64encode(row[1]).decode('utf-8') if row[1] else None,  # Convert BLOB to base64 string
            'probabilities': row[2],
            'prediction': row[3],
            'raw_text': row[4],
            'clean_text': row[5],
            'timestamp': row[6]
        })

    return jsonify({
        "status": "success",
        "data": data
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)
