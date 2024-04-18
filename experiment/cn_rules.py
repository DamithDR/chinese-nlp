import jieba

from experiment.metaphor import load_data, print_information

base_path = 'data/keywords/cn/'
noun_files = ["animals.txt",
              "body parts.txt",
              "emotion.txt",
              "gods and ghosts.txt",
              "kinship.txt",
              "manmade goods.txt",
              "natural elements.txt"]
verb_files = ['action_verbs.txt']

nouns = []
for f in noun_files:
    # print(f'{base_path}{f}')
    with open(f'{base_path}{f}', 'r') as fl:
        nouns.extend(fl.readlines())
nouns = [noun.strip().replace('\n', '') for noun in nouns]
#
# verbs = []
# for f in verb_files:
#     with open(f'{base_path}{f}', 'r') as fl:
#         verbs.extend(fl.readlines())

import jieba.posseg as pseg


# Function to find metaphoric flower names in a sentence
def find_metaphoric_flower_names(sentence):
    words = pseg.cut(sentence)

    metaphoric_flower_names = []
    for word, pos in words:
        if word in nouns:
            flower_name = word
            for next_word, next_pos in words:
                if next_pos == 'n':
                    flower_name += next_word
                else:
                    break
                metaphoric_flower_names.append(flower_name)

    return metaphoric_flower_names


# Iterate through the sentences and find metaphoric flower names
train_df, eval_df, tokenized_test_sentences, gold_tags, raw_sentences = load_data()
responses = []
for sentence in raw_sentences:
    flower_names = find_metaphoric_flower_names(sentence)
    responses.append(flower_names)
    if flower_names:
        print("Metaphoric flower names found in '{}': name : {}".format(sentence, ", ".join(flower_names)))

predictions = []
for tokenised_sentence, sentence, golds, metaphoric_names in zip(tokenized_test_sentences, raw_sentences, gold_tags,
                                                                 responses):
    predicted_tags = ['O'] * len(golds)
    for name in metaphoric_names:
        tokenised_flower = list(jieba.cut(name))
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
gold = [tag for tag_lst in gold_tags for tag in tag_lst]

print_information(pred, gold)
