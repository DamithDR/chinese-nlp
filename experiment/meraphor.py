import argparse
import json
import os.path

import pandas as pd
import torch
from simpletransformers.ner import NERModel
from sklearn.metrics import recall_score, precision_score, f1_score

from data.corpus_annotator import get_annotations
from sklearn.model_selection import train_test_split


# bert-base-chinese
#
#

def print_information(predictions, real_values):
    labels = set(real_values)

    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))


def format_data(dataset):
    sentence_ids = []
    words = []
    labels = []
    sent_id = 0
    dataset = pd.DataFrame(dataset)
    for sent, tags in zip(dataset['tokenised_sentence'], dataset['tags']):
        sent_index = [sent_id] * len(tags)
        sentence_ids.extend(sent_index)
        words.extend(sent)
        labels.extend(tags)
        sent_id += 1
    return pd.DataFrame({'sentence_id': sentence_ids, 'words': words, 'labels': labels})


def format_test_data(dataset):
    sentences = []
    tags_list = []
    dataset = pd.DataFrame(dataset)
    for sent, tags in zip(dataset['tokenised_sentence'], dataset['tags']):
        sentences.append(sent)
        tags_list.append(tags)
    return sentences, tags_list


def load_data():
    if os.path.exists('tagged_dataset.json'):
        with open('tagged_dataset.json', 'r') as file:
            dataset = json.load(file)
    else:
        dataset = get_annotations()
    train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=777)
    train, evaluation = train_test_split(train, test_size=0.1, shuffle=True, random_state=777)
    train_df = format_data(train)
    test_sentences, gold_tags = format_test_data(test)
    eval_df = format_data(evaluation)
    return train_df, eval_df, test_sentences, gold_tags


def run(args):
    train_df, eval_df, test_sentences, gold_tags = load_data()

    model = NERModel(
        model_type=args.model_type,
        model_name=args.model_name,
        labels=['B', 'I', 'O'],
        use_cuda=torch.cuda.is_available(),
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "num_train_epochs": 3,
              "train_batch_size": 32,
              },
    )

    model.train_model(train_df, eval_data=eval_df)

    predictions, raw_outputs = model.predict(test_sentences, split_on_space=False)

    flat_predictions = [list(tag.values()) for prediction in predictions for tag in prediction]
    flat_predictions = [p for pred in flat_predictions for p in pred]

    flat_gold_values = [value for gold_list in gold_tags for value in gold_list]

    print_information(flat_predictions, flat_gold_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--model_type', type=str, required=True, help='model_type')

    args = parser.parse_args()
    run(args)
