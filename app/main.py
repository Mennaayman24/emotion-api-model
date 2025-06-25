from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os
import requests

app = FastAPI()

# Google Drive model info
model_id = "1jGxqHGLDZ0lgBQ_9DW3pcumeuC-1XDdv"
model_url = f"https://drive.google.com/uc?export=download&id={model_id}"
model_path = "app/model/emotion-ferplus.onnx"

# Download model if not exists
if not os.path.exists(model_path):
    print("Model not found locally, downloading...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        response = requests.get(model_url)
        if response.status_code == 200:
            f.write(response.content)
            print("Model downloaded successfully.")
        else:
            raise RuntimeError("Failed to download model from Google Drive.")

# Load ONNX model
session = ort.InferenceSession(model_path)

# FER+ Labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.get("/")
def root():
    return {"message": "Emotion Recognition API is running"}

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L").resize((64, 64))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        img_np = np.array(image).astype(np.float32)
        img_np = np.expand_dims(img_np, axis=0)    # shape: (1, 64, 64)
        img_np = np.expand_dims(img_np, axis=0)    # shape: (1, 1, 64, 64)

        # Get model input name dynamically
        input_name = session.get_inputs()[0].name

        outputs = session.run(None, {input_name: img_np})
        print("Model output shape:", outputs[0].shape)

        pred = int(np.argmax(outputs[0]))
        emotion = emotion_labels[pred] if pred < len(emotion_labels) else "Unknown"

        return JSONResponse(content={"emotion": emotion})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
