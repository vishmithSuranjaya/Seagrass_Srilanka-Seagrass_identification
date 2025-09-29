from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import base64
import torch

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

    # Run YOLO inference
    results = model(img, conf=0.1)
    boxes = results[0].boxes

    if boxes.cls.numel() == 0:
        return jsonify({"error": "No objects detected"}), 400

    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    base_dim = max(img_width, img_height)
    scale = base_dim / 1000

    font_size = min(80, max(20, int(24 * scale)))
    box_width = min(16, max(4, int(6 * scale)))

    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except:
        font = ImageFont.load_default()

    # 🔹 Step 1: Organize detections by class
    detections_by_class = {}
    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        if cls_id not in detections_by_class or conf > detections_by_class[cls_id]["conf"]:
            detections_by_class[cls_id] = {
                "index": i,
                "conf": conf
            }

    # 🔹 Step 2: Annotate the best detection from each class
    predictions = []
    for cls_id, det in detections_by_class.items():
        i = det["index"]
        class_name = results[0].names[cls_id]
        confidence = det["conf"]

        x1, y1, x2, y2 = boxes.xyxy[i]
        box = [x1.item(), y1.item(), x2.item(), y2.item()]

        color = "white"  # You can generate per-class colors if needed
        draw.rectangle(box, outline=color, width=box_width)
        label = f"{class_name} ({confidence:.2f})"
        text_pos = (box[0], max(0, box[1] - font_size - 4))
        draw.text(text_pos, label, fill=color, font=font)

        predictions.append({
            "class": class_name,
            "confidence": f"{confidence:.2f}"
        })

    # 🔹 Step 3: Encode image
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    encoded_img_with_header = f"data:image/jpeg;base64,{encoded_img}"

    # 🔹 Step 4: Return JSON with all best detections (per class)
    return jsonify({
        "prediction": class_name,
        "confidence": f"{confidence:.2f}",
        "annotated_image": encoded_img_with_header
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
