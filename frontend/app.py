from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React frontend to communicate with this backend

@app.route('/run', methods=['GET'])
def run_application():
    # Replace this with the logic of your Python application
    result = {
        "category1": ["Output 1A", "Output 1B"],
        "category2": ["Output 2A", "Output 2B"],
        "category3": ["Output 3A", "Output 3B"],
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)