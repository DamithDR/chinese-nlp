# Load model directly
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from experiment.metaphor import load_data


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()

    # testing
    # dataset = dataset[100:102]
    # prompt_list = generate_prompts(dataset)
    # prompt_list = [f'<s>[INST] {prompt} [/INST]'for prompt in prompt_list]

    generation_config = GenerationConfig(
        max_new_tokens=100, do_sample=True, top_k=20, eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id, temperature=0.2,
        num_return_sequences=1, torch_dtype=torch.bfloat16,
    )

    prompt = """
        A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
        Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
        For example, the plant name "Corn" in Chinese is "玉蜀黍" (yù shǔ shǔ), where "玉" means jade. The word "jade" does not describe an attribute of the plant itself, but a characteristic of the plant is similar to a quality of jade, thus "jade" is used metaphorically to imply the plant is as precious or as beautiful as jade. Therefore, this plant name can be determined as a metaphorical plant name.
        Your task is to identify metaphorical flower names in the given chinese text.
        You must identify if there are metaphoric flower names respond as a json format of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
        Example json object : {metaphoric_names_found : 'yes',metaphoric_names = ['name1','name2']}
        Do not provide any other explanation. Just return json object with the results.

        Sentence : """

    outputs = []
    for sentence, gold_tags in zip(raw_sentences, gold_tags):
        prompt += sentence

        encoding = tokenizer(prompt, padding=True, truncation=False, return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                 generation_config=generation_config)
        detach = outputs.detach().cpu().numpy()
        outputs = detach.tolist()
    out_list = []
    out_list.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])

    alias = str(args.model_name).replace('/', '_')
    with open(f'raw_results_from_model{alias}.txt', 'w') as f:
        f.writelines("\n".join(out_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')

    args = parser.parse_args()
    run(args)
