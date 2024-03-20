import json

import pandas as pd

from experiment.metaphor import load_data

english = 'chatgpt_responses.json'
chinese = 'chatgpt_responses_chinese.json'

train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()

responses = []

print(f'Total no of sentences : {len(raw_sentences)}')

with open(english, 'r') as e_f:
    english_data_list = json.load(e_f)

with open(chinese, 'r') as c_f:
    chinese_data_list = json.load(c_f)

df = pd.DataFrame({"text": raw_sentences, "english_response": english_data_list, "chinese_response": chinese_data_list})

df.to_excel("results.xlsx", sheet_name="results")
