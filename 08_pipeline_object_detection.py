# Import the pipeline function from the transformers library
from transformers import pipeline 

# Load the object detection pipeline
detector = pipeline(task="object-detection")

# Detect objects in an image
preds = detector(
    "https://e00-marca.uecdn.es/assets/multimedia/imagenes/2025/04/20/17451314384383.jpg"
)

# Round the scores to 4 decimal places and keep the label and box coordinates for each prediction
preds = [{
    "score": round(pred["score"], 4),
    "label": pred["label"],
    "box": pred["box"]
} for pred in preds]

# Print the predictions
print(preds) 