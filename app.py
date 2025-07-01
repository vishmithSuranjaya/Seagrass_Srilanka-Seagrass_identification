from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import base64
import random

app = Flask(__name__)
CORS(app)

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

    detected_classes = [int(cls.item()) for cls in boxes.cls]
    all_same = len(set(detected_classes)) == 1

    species_colors = {}
    if all_same:
        default_color = "lime"
    else:
        for cls_id in set(detected_classes):
            species_colors[cls_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i].item())
        class_name = results[0].names[cls_id]
        confidence = float(boxes.conf[i].item())

        x1, y1, x2, y2 = boxes.xyxy[i]
        box = [x1.item(), y1.item(), x2.item(), y2.item()]

        if all_same:
            color = default_color
        else:
            r, g, b = species_colors[cls_id]
            color = f"#{r:02x}{g:02x}{b:02x}"

        draw.rectangle(box, outline=color, width=box_width)
        label = f"{class_name} ({confidence:.2f})"
        text_pos = (box[0], max(0, box[1] - font_size - 4))
        draw.text(text_pos, label, fill=color, font=font)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    encoded_img_with_header = f"data:image/jpeg;base64,{encoded_img}"

    first_class_name = results[0].names[int(boxes.cls[0].item())]
    first_confidence = float(boxes.conf[0].item())

    return jsonify({
        "prediction": first_class_name,
        "confidence": f"{first_confidence:.2f}",
        "annotated_image": encoded_img_with_header
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
