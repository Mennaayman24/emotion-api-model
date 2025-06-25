from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os

app = FastAPI()

# Load the ONNX model
model_path = "app/model/resnet34-v2-7.onnx"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")

session = ort.InferenceSession(model_path)
emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

@app.get("/")
def root():
    return {"message": "Emotion Recognition API is running"}

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L").resize((64, 64))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_np = np.array(image).astype(np.float32)
    img_np = np.expand_dims(img_np, axis=0)  # 1x64x64
    img_np = np.expand_dims(img_np, axis=0)  # 1x1x64x64

    try:
        outputs = session.run(None, {"Input3": img_np})
        pred = np.argmax(outputs[0])
        emotion = emotion_labels[pred]
        return JSONResponse(content={"emotion": emotion})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
