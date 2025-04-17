# Import the pipeline function from the transformers library
from transformers import pipeline

# Load the sentiment analysis pipeline with a specific model
classifier = pipeline("sentiment-analysis")

# Classify the sentiment of a given text
response = classifier("Creo que tienes potencial para ser un gran programador")

# Print the classification result
print(response)

