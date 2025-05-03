# Import the pipeline function from the transformers library
from transformers import pipeline 

# Create a token classification pipeline for named entity recognition (NER)
classifier = pipeline(task="ner")

# Define a sample text for named entity recognition
text = "Google is a technology company based in Mountain View, California."

# Use the classifier to predict named entities in the text
preds = classifier(text)

# Format the predictions to include only relevant fields
preds = [
    {
        "entity": pred["entity"],
        "score": round(pred["score"], 4),
        "index": pred["index"],
        "word": pred["word"],
        "start": pred["start"],
        "end": pred["end"],
    }
    for pred in preds
]

# Print the formatted predictions
print(*preds, sep="\n")