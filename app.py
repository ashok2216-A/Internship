from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import re
import string
import numpy as np

# 1. Load model and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 2. FastAPI app
app = FastAPI()

# 3. Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text

# 4. Input schema
class MessageInput(BaseModel):
    message: str

# 5. Endpoint
@app.post("/categorize")
def categorize(input_data: MessageInput):
    cleaned = clean_text(input_data.message)
    pred_probs = model.predict_proba([cleaned])[0]
    pred_label_idx = np.argmax(pred_probs)
    pred_category = label_encoder.inverse_transform([pred_label_idx])[0]
    confidence = round(float(pred_probs[pred_label_idx]), 4)

    return {
        "category": pred_category,
        "confidence": confidence
    }

# 6. Main
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


