import json

import jieba
import pandas as pd

flower_names = pd.read_csv('data/processed/annotation.tsv', sep='\t')

metaphoric_set = set(flower_names[flower_names['label'] == 1]['name'].tolist())

print(metaphoric_set)
print(len(metaphoric_set))

with open('data/processed/sentenses.txt', 'r') as corpus:
    sentences = corpus.readlines()

    sentences = [sentence.replace('\n', '').strip() for sentence in sentences]
    final_dataset = []
    for sentence in sentences:
        tokenised_sentence = list(jieba.cut(sentence))
        tags = ['O'] * len(tokenised_sentence)  # tagging everything as O
        for flower_name in metaphoric_set:
            tokenised_flower = list(jieba.cut(flower_name))

            if tokenised_flower[0] in tokenised_sentence:
                first_index = tokenised_sentence.index(tokenised_flower[0])
                found = True
                if first_index + len(tokenised_flower) <= len(tokenised_sentence):
                    for i in range(1, len(tokenised_flower)):
                        if tokenised_flower[i] != tokenised_sentence[first_index + i]:
                            found = False
                    if found:
                        tags[first_index] = 'B'
                        for i in range(first_index + 1, first_index + len(tokenised_flower)):
                            tags[i] = 'I'
                        print(
                            f'found first word match : {tokenised_flower} \n {tokenised_sentence} \n {tags} \n ==================')
            data_dict = {}
            data_dict['tokenised_sentence'] = tokenised_sentence
            data_dict['tags'] = tags
            data_dict['sentence'] = sentence
            final_dataset.append(data_dict)

    with open('tagged_dataset.json', 'w', ) as json_file:
        json.dump(final_dataset, json_file, ensure_ascii=False)
