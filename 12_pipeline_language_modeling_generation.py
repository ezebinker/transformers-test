# Import the pipeline function from the transformers library
from transformers import pipeline 

# Save the prompt 
prompt = "Hugging Face is a community-based open-source platform for machine learning."

# Create a text generation pipeline
generator = pipeline(task="text-generation")

# Use the generator to generate text based on the prompt
text = generator(prompt)

# Print the generated text
print(*text, sep="\n")