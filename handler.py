import runpod
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import base64
import io

# Load model once on cold start
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()


def handler(event):
    # Input format: {"image_base64": "...", "text": "cat", "box_threshold": 0.3}
    image_b64 = event["image_base64"]
    text = event.get("text", "object")
    box_threshold = float(event.get("box_threshold", 0.3))

    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

    # Prepare inputs
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Model inference
    outputs = model(**inputs)

    # Post-processing
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=0.25
    )[0]

    return {
        "boxes": results["boxes"].tolist(),
        "labels": results["labels"],
        "scores": results["scores"].tolist()
    }


runpod.serverless.start({"handler": handler})
