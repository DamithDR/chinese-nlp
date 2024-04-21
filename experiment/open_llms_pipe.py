import argparse
import json
import re

import torch.cuda
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from experiment.metaphor import load_data
from experiment.open_llms import get_template


class ListDataset:

    def __init__(self, data_list):
        self.original_list = data_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def extract_json_objects(string):
    pattern = r'{(.*?)}'  # Matches anything between curly braces
    json_objects = re.findall(pattern, string)
    json_objects = ["{" + obj + "}" for obj in json_objects]
    if json_objects:
        return json_objects
    else:
        return [{"metaphoric_names_found": "no", "metaphoric_names": []}]


def run(args):
    # model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    text_generator = pipeline(
        "text-generation",  # task
        tokenizer=tokenizer,
        model=args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
        do_sample=True,
        max_new_tokens=50,
        batch_size=args.batch_size,
        top_k=20,
        top_p=0.8,
        num_return_sequences=1,
        temperature=0.1,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id,
    )

    # Data
    # chinese
    raw_sentences = []
    if args.language == 'zh':
        train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()
    elif args.language == 'en':
        with open('data/en_es/en.txt', 'r') as f:
            raw_sentences = f.readlines()
            raw_sentences = [sent.replace('\n', '') for sent in raw_sentences]
    elif args.language == 'es':
        with open('data/en_es/es.txt', 'r') as f:
            raw_sentences = f.readlines()
            raw_sentences = [sent.replace('\n', '') for sent in raw_sentences]

    raw_sentences = raw_sentences[:20]

    template = get_template(args.language)
    prompts = list(map(lambda sent: template + sent, raw_sentences))
    prompts = list(map(lambda prompt: '[INST]' + prompt + '[/INST]', prompts))

    print(f'prompt 1 : {prompts[0]}')
    print(f'prompt 2 : {prompts[1]}')
    print(f'prompt 3 : {prompts[2]}')

    print('predicting outputs...')
    results = text_generator(prompts)
    print('predicting completed...')

    # testing
    results = [result[0]['generated_text'].split('[/INST]')[1].strip() for result in results]
    results = [extract_json_objects(result)[0] for result in results]

    print(f'results 1 : {results[0]}')
    print(f'results 2 : {results[1]}')
    print(f'results 3 : {results[2]}')

    print('json loading started')
    objects = [json.loads(result) for result in tqdm(results)]
    print('json loading finished')

    alias = str(args.model_name).replace('/', '_')

    with open(f'{alias}_{args.language}_results.json', "w") as json_file:
        json.dump(objects, json_file, ensure_ascii=False)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='batch_size')
    parser.add_argument('--language', type=str, required=True, default='zh', help='language to run experiment')

    args = parser.parse_args()
    run(args)
