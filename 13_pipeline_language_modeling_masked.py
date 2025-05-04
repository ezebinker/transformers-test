# Import the pipeline function from the transformers library
from transformers import pipeline 

# Save the prompt 
text = "<mask> es la capital de España."

# Create a fill-mask pipeline
fill_mask = pipeline(task="fill-mask", model="bertin-project/bertin-roberta-base-spanish")

# Use the fill-mask pipeline to predict the masked token in the text
preds = fill_mask(text, top_k=1)

# Format the predictions to include only relevant fields
preds = [
    {
        "score": round(pred["score"], 4),
        "token": pred["token"],
        "token_str": pred["token_str"],
        "sequence": pred["sequence"],
    }
    for pred in preds
]

# Print the formatted predictions
print(*preds, sep="\n")