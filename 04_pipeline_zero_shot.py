# Import transformers library
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define the sequence to classify
sequence = "15 minutes and Ronaldo has the ball in the middle of the field."

# Define the candidate labels
candidate_labels = ["education", "politics", "business", "sports"]

# Perform zero-shot classification
result = classifier(sequence, candidate_labels)

# Print the result
print(result)