import base64
import io
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ---------------------------
# Load model at startup
# ---------------------------
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI(title="GroundingDINO Server", version="1.0")


# ---------------------------
# Request model
# ---------------------------
class InferenceRequest(BaseModel):
    image_base64: str
    text: str = "object"
    box_threshold: float = 0.3


# ---------------------------
# /infer endpoint
# ---------------------------
@app.post("/infer")
def infer(req: InferenceRequest):

    # Decode Base64 → PIL image
    image = Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB")

    # Preprocess
    inputs = processor(
        images=image,
        text=req.text,
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=req.box_threshold,
        text_threshold=0.25
    )[0]

    # Convert tensors to lists
    return {
        "boxes": results["boxes"].cpu().tolist(),
        "labels": results["labels"],
        "scores": results["scores"].cpu().tolist()
    }


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok"}
