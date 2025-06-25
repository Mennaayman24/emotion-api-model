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
model_id = "10Cyx2vJ1xYofRb44NV0JVwewBKnbDAod"
model_url = f"https://drive.google.com/uc?export=download&id={model_id}"
model_path = "app/model/resnet34-v2-7.onnx"

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
emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

@app.get("/")
def root():
    return {"message": "Emotion Recognition API is running"}

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Open and resize image to 224x224, and convert to RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Convert image to numpy array and prepare it for the model
        img_np = np.array(image).astype(np.float32) / 255.0  # Normalize
        img_np = np.transpose(img_np, (2, 0, 1))             # From HWC to CHW
        img_np = np.expand_dims(img_np, axis=0)              # Add batch dimension: [1, 3, 224, 224]

        outputs = session.run(None, {"data": img_np})

        print("Model outputs:", outputs)

        if not outputs or outputs[0].size == 0:
            raise HTTPException(status_code=500, detail="Model returned empty output.")

        pred = int(np.argmax(outputs[0]))
        emotion = emotion_labels[pred]
        return JSONResponse(content={"emotion": emotion})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
