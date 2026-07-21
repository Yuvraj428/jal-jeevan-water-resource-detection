import base64
import io
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO 


MODEL_PATH = "../runs/jjh_yolov8s/weights/best.pt"  
CONF_THRESHOLD = 0.25                                
IOU_THRESHOLD = 0.5
IMG_SIZE = 640
DEVICE = "cpu"

CLASS_NAMES = ['handpump', 'lake', 'pine', 'pine_dry', 'well']


DISPLAY_NAME_MAP = {
    "pine_dry": "pine",
}


DISPLAY_NAMES_BY_INDEX = {
    i: DISPLAY_NAME_MAP.get(name, name) for i, name in enumerate(CLASS_NAMES)
}


app = Flask(__name__)

print("[INFO] Loading YOLOv8 model... this may take a few seconds.")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")



def decode_base64_image(b64_string):


    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Check the base64 string is valid.")
    return img


def encode_image_to_base64(img):
  
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Could not encode image to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")



@app.route("/", methods=["GET"])
def health_check():

    return jsonify({"status": "ok", "message": "JJH YOLOv8 API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if data is None or "image" not in data:
            return jsonify({"error": "Request body must contain an 'image' (base64) field."}), 400

        image_id = data.get("image_id", "unknown")
        b64_image = data["image"]

        img = decode_base64_image(b64_image)

        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )
        result = results[0]

        predictions = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            internal_name = CLASS_NAMES[cls_id]
            display_name = DISPLAY_NAME_MAP.get(internal_name, internal_name)

            predictions.append({
                "class": display_name,
                "confidence": round(conf, 4)
            })

        result.names = DISPLAY_NAMES_BY_INDEX
        annotated_img = result.plot()

        annotated_b64 = encode_image_to_base64(annotated_img)

        response = {
            "image_id": image_id,
            "predictions": predictions,
            "annotated_image": annotated_b64
        }
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=5000, debug=True)