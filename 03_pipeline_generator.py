# Import the pipeline function from the transformers library
from transformers import pipeline

# Load the text generation pipeline with a specific model
generator = pipeline("text-generation", model="openai-community/gpt2-large")

# Generate text based on a prompt
response = generator("Manchester United is a team of great", max_length=50, num_return_sequences=2)

# Print the generated text
print(response)