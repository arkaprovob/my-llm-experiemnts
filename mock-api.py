from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


# Function to log request details
def log_request(req: request):
    app.logger.debug('Request Headers: %s', req.headers)
    app.logger.debug('Request Body: %s', req.get_data(as_text=True))


# Route for both GET and POST requests
@app.route('/api/generate/', methods=['GET', 'POST'])
def generate():
    # Log the request
    log_request(request)

    if request.method == 'GET':
        # Handle GET request
        return jsonify({"message": "This is a GET request", "data": "You can place your data here"})
    elif request.method == 'POST':
        # Handle POST request
        data = request.json  # Assuming JSON data is sent in POST request
        return jsonify({"message": "This is a POST request", "received_data": data})


if __name__ == '__main__':
    app.run(port=11434)
