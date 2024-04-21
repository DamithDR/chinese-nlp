from transformers import pipeline

# Define the model name or path
model_name = "meta-llama/Llama-2-13b-chat-hf"

# Create the text generation pipeline
text_generator = pipeline("text-generation", model=model_name)

# Generate text using the pipeline
text = text_generator("How are you", max_length=50, do_sample=True)[0]['generated_text']

print(text)
