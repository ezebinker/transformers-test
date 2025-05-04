# Import the pipeline function from the transformers library
from transformers import pipeline 

# Import the Image class from the PIL library
from PIL import Image

# Import the requests library for making HTTP requests
import requests

# URL of the image to be used in the document
image_url = "https://datasets-server.huggingface.co/assets/katanaml-org/invoices-donut-data-v1/--/default/train/16/image/image.jpg?Expires=1746388076&Signature=fsAOAgCoJDukZ7VkKiGnHP7fZa0mABMbaXEUuIQaPVVO3ox5naxgfPO6YVC7FEXPWpA5CF6yA9IQ3NexN92iOs6bMxqiXJAfrP0gU87g9IEDGs6PDdXgOAQp3KKl87SPCIIhig3U~Pm4YQdIoEm9Feqt-uETCJ1Dfnzpq2vCznVU~GFjHrrwSX0809SkwY7xN7hvVZEkt4WgV-AXru6u9SCFdUbmlzEzRileSIg2GbPpYDUbVtxcLPoNKraEdQklI5Qg~E-Qe4i9U~uodzQraR-IJeGN~J~7I0~Hsg4r5B7vpigOk~upEbw3OEbKQeIaLvVZwLcOfqvJNstZjX~hVw__&Key-Pair-Id=K3EI6M078Z3AC3"

# Download the image from the URL
image = Image.open(requests.get(image_url, stream=True).raw)

# Create a multimodal document question-answering pipeline
question_answerer = pipeline(task="document-question-answering", model="magorshunov/layoutlm-invoices")

# Define a sample question to be asked about the document
question = "What is the gross worth of the invoice?"

# Use the question-answering pipeline to get the answer to the question
preds = question_answerer(question=question, image=image)

# Print the predictions
print(preds)
