# Import the pipeline function from the transformers library
from transformers import pipeline 

# Load the audio classification pipeline
classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
# The model is a HuBERT model fine-tuned on the SUPERB dataset for audio classification tasks.

# Classify an audio file
preds = classifier(
    "https://www.myinstants.com/media/sounds/tu-tu-tu-du-max-verstappen.mp3"
)

# Round the scores to 4 decimal places and keep the label for each prediction
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

# Print the predictions
print(preds)