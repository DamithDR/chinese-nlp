import torch.cuda
from transformers import pipeline

# Define the model name or path
model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-hf"
# model_name = "tiiuae/falcon-40b"

# Create the text generation pipeline
# text_generator = pipeline("text-generation", model=model_name,device_map='auto')
text_generator = pipeline(
    "text-generation",  # task
    model=model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    do_sample=True,
    max_new_tokens=100,
    # batch_size=args.batch_size,
    top_k=50,
    top_p=0.5,
    num_return_sequences=2,
    temperature=0.1,
    # eos_token_id=tokenizer.eos_token_id,
    # pad_token_id=tokenizer.eos_token_id,
)

# Generate text using the pipeline

prompt = """
    A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
    Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
    In other words, metaphoric flower name is one that uses a word or phrase that evokes certain qualities, characteristics, or associations beyond the literal description of the flower itself.
    For example, the flower name "Forget-me-not" is metaphorical. While it directly refers to the tiny, delicate blue flowers of the Myosotis genus, its name carries emotional connotations beyond its physical appearance. "Forget-me-not" suggests remembrance, loyalty, and enduring love. It metaphorically implies that the giver of the flower is asking the recipient not to forget them and to remember the bond they share, making it a poignant and symbolic name for this charming flower.
    Your task is to identify metaphorical flower names in the given English text.
    You must identify if there are metaphoric flower names respond as a json object of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
    Example json object : {metaphoric_names_found : 'yes',metaphoric_names = ['name1','name2']}
    Do not provide any other explanation. Just return json object with the results.

    Sentence : See ceanothus ‘Gloire de Versailles’
"""

# prompt = """
#     A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
#     Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
#     In other words, metaphoric flower name is one that uses a word or phrase that evokes certain qualities, characteristics, or associations beyond the literal description of the flower itself.
#     For example, the flower name "Forget-me-not" is metaphorical. While it directly refers to the tiny, delicate blue flowers of the Myosotis genus, its name carries emotional connotations beyond its physical appearance. "Forget-me-not" suggests remembrance, loyalty, and enduring love. It metaphorically implies that the giver of the flower is asking the recipient not to forget them and to remember the bond they share, making it a poignant and symbolic name for this charming flower.
#     Your task is to identify metaphorical flower names in the given English text.
#     You must identify if there are metaphoric flower names respond yes/no (depending on there is a metaphorical flower name or not) and the list of names liseted as "names_list".
#     Example response : 'yes', names_list = ['name1','name2']}
#
#     Sentence : See ceanothus ‘Gloire de Versailles’
# """

print(torch.cuda.device_count())

print("===========================================================================")

text = text_generator(prompt)[0]['generated_text']

print(text)
