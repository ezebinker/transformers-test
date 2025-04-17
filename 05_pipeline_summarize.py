from transformers import pipeline # Import the pipeline function from the transformers library
from datasets import load_dataset # Import the datasets library to load the IMDB dataset
import json # Import the json library to save results
import os # Import the os library to handle file operations

# Load the dataset
dataset = load_dataset("imdb")

# Load the summarization pipeline
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text):
    # Use the summarization pipeline to summarize the text
    summary = summarization_pipeline(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Save the results to a json array file
output_file = "output/summarization_results.json"
if os.path.exists(output_file):
    os.remove(output_file)

# Create a directory for the output file if it doesn't exist
results = []
for i in range(5):
    review = dataset['test'][i]['text']
    summary = summarize_text(review)
    result = {
        "review": review,
        "summary": summary
    }
    results.append(result)

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the results to a json file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)