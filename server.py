from flask import Flask, request, send_file, jsonify
import io
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model once when the server starts
model = YOLO("weights/best.pt")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the image file into a numpy array
        in_memory_file = io.BytesIO(file.read())
        image = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Perform prediction with confidence threshold 0.6
        results = model.predict(source=image, conf=0.6, save=False, verbose=False)

        # Draw red outlines on the image
        for result in results:
            for mask in result.masks.xy:  # get polygon coordinates
                mask = np.array(mask, np.int32)  # convert them to integer
                mask = mask.reshape((-1, 1, 2))  # reshape required for drawing the red outline
                cv2.polylines(image, [mask], isClosed=True, color=(0, 0, 255), thickness=2)

        # Encode the image to a byte stream
        _, buffer = cv2.imencode('.jpg', image)
        byte_io = io.BytesIO(buffer)

        return send_file(byte_io, mimetype='image/jpeg', as_attachment=True, download_name='output_image.jpg')

if __name__ == "__main__":
    app.run(debug=True)