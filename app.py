from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import base64

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']

    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {str(e)}"}), 400

    # Run inference
    results = model(img, conf=0.1)
    boxes = results[0].boxes

    if boxes.cls.numel() == 0:
        return jsonify({"error": "No objects detected"}), 400

    draw = ImageDraw.Draw(img)

    # Optional: load a truetype font (use default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except:
        font = ImageFont.load_default()

    # Draw all detections
    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i].item())
        class_name = results[0].names[cls_id]
        confidence = float(boxes.conf[i].item())

        # Get coordinates
        x1, y1, x2, y2 = boxes.xyxy[i]
        box = [x1.item(), y1.item(), x2.item(), y2.item()]
        
        # Draw bounding box
        draw.rectangle(box, outline="white", width=10)

        # Draw label
        label = f"{class_name} ({confidence:.2f})"
        draw.text((box[0], box[1] - 10), label, fill="white", font=font)

    # Convert annotated image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    encoded_img_with_header = f"data:image/jpeg;base64,{encoded_img}"

    # Return first prediction + image
    first_class_name = results[0].names[int(boxes.cls[0].item())]
    first_confidence = float(boxes.conf[0].item())

    return jsonify({
        "prediction": first_class_name,
        "confidence": f"{first_confidence:.2f}",
        "annotated_image": encoded_img_with_header
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
