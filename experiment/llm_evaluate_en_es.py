import argparse
import json

import jieba
import pandas as pd

from experiment.metaphor import load_data, print_information


def run(args):
    # train_df, eval_df, tokenized_test_sentences, gold_tags, raw_sentences = load_data()
    if args.language == 'en':
        with open('data/en_es/en.txt', 'r') as f:
            sentences = f.readlines()
            sentences = [sent.replace('\n', '') for sent in sentences]
        golds = pd.read_csv('data/en_es/en_test.tsv', sep='\t')
        golds = golds['labels'].to_list()
    elif args.language == 'es':
        golds = pd.read_csv('data/en_es/es_test.tsv', sep='\t')
        golds = golds['labels'].to_list()
        with open('data/en_es/es.txt', 'r') as f:
            sentences = f.readlines()
            sentences = [sent.replace('\n', '') for sent in sentences]

    if args.file_name:
        with open(args.file_name, 'r') as file:
            responses = json.load(file)
    else:
        raise ValueError('file name not set')

    predictions = []
    for sentence, response in zip(sentences, responses):
        tokenised_sentence = sentence.split(" ")
        predicted_tags = ['O'] * len(tokenised_sentence)
        if str(response['metaphoric_names_found']).lower() in ['yes']:
            if 'metaphoric_names' in response:
                metaphoric_names = response['metaphoric_names']
                for name in metaphoric_names:
                    tokenised_flower = name.split(" ")
                    if tokenised_flower[0] in tokenised_sentence:
                        first_index = tokenised_sentence.index(tokenised_flower[0])
                        found = True
                        if first_index + len(tokenised_flower) <= len(tokenised_sentence):
                            for i in range(1, len(tokenised_flower)):
                                if tokenised_flower[i] != tokenised_sentence[first_index + i]:
                                    found = False
                            if found:
                                predicted_tags[first_index] = 'B'
                                for i in range(first_index + 1, first_index + len(tokenised_flower)):
                                    predicted_tags[i] = 'I'
                                print(
                                    f'found first word match : {tokenised_flower} \n {tokenised_sentence} \n {predicted_tags} \n ==================')
        predictions.append(predicted_tags)

    pred = [token for prediction in predictions for token in prediction]

    print_information(pred, golds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluate llms on their output''')
    parser.add_argument('--file_name', type=str, required=True, help='response_file')
    parser.add_argument('--language', type=str, default='es', required=False, help='response_file')

    args = parser.parse_args()
    run(args)
