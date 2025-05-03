# Import the pipeline function from the transformers library
from transformers import pipeline 

# Create a pipeline for image segmentation
segmenter = pipeline(task="image-segmentation")

# Segment an image
preds = segmenter("https://media-cdn.tripadvisor.com/media/attractions-splice-spp-674x446/0e/3b/66/8f.jpg")

# Round the scores to 4 decimal places and keep the label for each prediction
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

# Print the predictions
print(preds)