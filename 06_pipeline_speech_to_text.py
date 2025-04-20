# Import the pipeline function from the transformers library
from transformers import pipeline 

# Load the transcription pipeline
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

# Transcribe an audio file
transcription = transcriber("https://datasets-server.huggingface.co/cached-assets/ylacombe/expresso/--/9fb79a189698de3255eff48edd2bc0d9e487adc0/--/read/train/3/audio/audio.wav?Expires=1745181453&Signature=wTlgI9RHNVb24UXVhL7M3uEQUR~rGKQdricOqnhPJkXkb1Zz4bELYYt3hdkTY~3regcFdaA0zQVFp3l3V7ULpOimieiq55Hc3lco0pk18NfYhDYoVLpgWd7ebHqdrOoxaW43mT1-spOYfoMFNa1w2qKYpwxVOp86glls~~3GA2p4YG5V1MpyTZyQ6KzdUINhnWy6THZEVbhyuuGiZz9QPUND2~dPT9nLfxFgKHrNdnMoR1YixW36BNQ30SLU7zL6393~4sZ1LjvO9xzsXoOHRsoWV67a-hMXSIn5APgeMo3IgyvL~tYkBcy~D13alSnWeC4n871hsnrzgK9pUVoxHQ__&Key-Pair-Id=K3EI6M078Z3AC3")

# Print the transcription
print(transcription)