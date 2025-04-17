from transformers import pipeline # Import the pipeline function from the transformers library
from datasets import load_dataset # Import the datasets library to load the IMDB dataset
import json # Import the json library to save results
import os # Import the os library to handle file operations

# Load the dataset
dataset = load_dataset("imdb")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label'], result[0]['score']

# Save the results to a json array file
output_file = "output/sentiment_results.json"
if os.path.exists(output_file):
    os.remove(output_file)

# Create a directory for the output file if it doesn't exist
results = []
for i in range(5):
    review = dataset['test'][i]['text']
    sentiment, score = analyze_sentiment(review)
    result = {
        "review": review,
        "sentiment": sentiment,
        "score": score
    }
    results.append(result)

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the results to a json file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)