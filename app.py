import os, json, base64
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plant_disease_mobilenetv2.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
INFO_PATH = ROOT / "data" / "disease_info.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
_model = None
_labels = None
_model_error = None

def load_model():
    global _model, _labels, _model_error
    if _model is not None:
        return _model, _labels, None
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None, "Model files are not available yet. Add labelled training images and run training/train_model.py."
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        return _model, _labels, None
    except Exception as exc:
        _model_error = str(exc)
        return None, None, "Could not load the trained model: " + _model_error

def load_info():
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/status")
def status():
    _, _, error = load_model()
    return jsonify({"model_ready": error is None, "message": "Model ready" if error is None else error})

@app.post("/api/predict")
def predict():
    model, labels, error = load_model()
    if error:
        return jsonify({"error": error}), 503
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    if not image_data:
        return jsonify({"error": "No camera image received."}), 400
    try:
        raw = base64.b64decode(image_data)
        img = Image.open(BytesIO(raw)).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        # MobileNetV2 preprocessing is included in the saved model.
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
        if confidence < threshold:
            return jsonify({
                "identified": False, "disease": "Unknown / Uncertain",
                "confidence": round(confidence * 100, 2),
                "message": "The model is not confident enough. Try a clear, well-lit image of one leaf."
            })
        label = labels[idx]
        info = load_info().get(label, {})
        return jsonify({
            "identified": True, "disease": label.replace("___", " - ").replace("_", " "),
            "confidence": round(confidence * 100, 2),
            "plant": info.get("plant", label.split("___")[0].replace("_", " ")),
            "symptoms": info.get("symptoms", "Disease-specific symptoms have not been added to the information file yet."),
            "preventions": info.get("preventions", ["Consult a local agricultural expert for verified advice."]),
            "note": "AI prediction only. Confirm uncertain cases with an agricultural expert."
        })
    except Exception as exc:
        return jsonify({"error": "Image processing failed: " + str(exc)}), 400

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    language = data.get("language", "en")
    context = data.get("context") or {}
    if not message:
        return jsonify({"error": "Please enter or speak a question."}), 400

    # Optional OpenAI-compatible API (OpenAI, Groq, or another compatible provider).
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or None
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            lang_name = "Telugu" if language == "te" else "English"
            system = (
                "You are a cautious agricultural information assistant. Answer in " + lang_name +
                ". Use the supplied prediction context when relevant. Do not claim certainty from an image classifier. "
                "Do not prescribe pesticide brands, dosages, or chemical mixtures. Give general integrated pest-management "
                "guidance and recommend local agricultural extension advice for treatment decisions. If the context is empty, "
                "ask for crop and symptoms rather than inventing a diagnosis. Keep answers practical and concise."
            )
            result = client.chat.completions.create(
                model=model_name, temperature=0.3,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":"Prediction context: "+json.dumps(context, ensure_ascii=False)+"\nQuestion: "+message}]
            )
            answer = result.choices[0].message.content
            return jsonify({"answer": answer, "source": "LLM"})
        except Exception as exc:
            return jsonify({"answer": fallback_answer(language), "warning": "LLM request failed: " + str(exc)})
    return jsonify({"answer": fallback_answer(language), "warning": "LLM API is not configured. Add LLM settings to .env for open-ended conversation.", "source":"fallback"})

def fallback_answer(language):
    if language == "te":
        return "మీ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి పంట పేరు, కనిపిస్తున్న లక్షణాలు మరియు ప్రభావితమైన ఆకుల వివరాలు అవసరం. కెమెరా అంచనాను మాత్రమే ఆధారంగా చేసుకుని మందులు వాడకండి; స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి."
    return "To give a reliable answer, I need the crop name and visible symptoms. Treat the camera result as an AI estimate, not a confirmed diagnosis. Please consult a local agricultural expert before choosing any treatment."

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
