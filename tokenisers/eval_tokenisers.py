import jieba
import thulac
import stanfordnlp
import pynlpir
from pyltp import Segmentor

import pandas as pd

df = pd.read_excel('data/data_file.xlsx', sheet_name='Com_species_corpus_500')
sampled_df = df.sample(n=20, random_state=42)

thu = thulac.thulac(seg_only=True)

stanfordnlp.download('zh')
nlp = stanfordnlp.Pipeline(lang='zh', processors='tokenize')

pynlpir.open(encoding='utf-8')
jieba_results_name = []
jieba_results_sentence = []
thu_results_name = []
thu_results_sentence = []
stanford_results_name = []
stanford_results_sentence = []
pynlpir_results_name = []
pynlpir_results_sentence = []
ltp_results_name = []
ltp_results_sentence = []

model_path = """D:\\chinese\\tokeniser models\\ltp_data_v3.4.0\\ltp_data_v3.4.0\\cws.model"""
segmentor = Segmentor(model_path)
# segmentor.load_with_lexicon(model_path + '/cws.model')


for name, sentence in zip(sampled_df['name'], sampled_df['sentence']):
    # jieba
    jieba_name = list(jieba.cut(name))
    jieba_results_name.append(jieba_name)
    jieba_tokens = list(jieba.cut(sentence))
    jieba_results_sentence.append(jieba_tokens)

    # thu
    thu_name = list(thu.cut(name, text=True))
    thu_results_name.append(thu_name)
    thu_tokens = list(thu.cut(sentence, text=True))
    thu_results_sentence.append(thu_tokens)

    # stanford
    doc = nlp(name)
    stanford_name = [word.text for sent in doc.sentences for word in sent.words]
    stanford_results_name.append(stanford_name)
    doc = nlp(sentence)
    stanford_tokens = [word.text for sent in doc.sentences for word in sent.words]
    stanford_results_sentence.append(stanford_tokens)

    # nlpir
    pynlpir_name = pynlpir.segment(name, pos_tagging=False)
    pynlpir_results_name.append(pynlpir_name)
    pynlpir_tokens = pynlpir.segment(sentence, pos_tagging=False)
    pynlpir_results_sentence.append(pynlpir_tokens)

    # ltp
    ltp_name = list(segmentor.segment(name))
    ltp_results_name.append(ltp_name)
    ltp_tokens = segmentor.segment(sentence)
    ltp_results_sentence.append(ltp_tokens)

pynlpir.close()
segmentor.release()

names_df = pd.DataFrame()
names_df['name'] = sampled_df['name']
names_df['jieba'] = jieba_results_name
names_df['thu'] = thu_results_name
names_df['stanford'] = stanford_results_name
names_df['nlpir'] = pynlpir_results_name
names_df['ltp'] = ltp_results_name

sentence_df = pd.DataFrame()
sentence_df['sentence'] = sampled_df['sentence']
sentence_df['jieba'] = jieba_results_sentence
sentence_df['thu'] = thu_results_sentence
sentence_df['stanford'] = stanford_results_sentence
sentence_df['nlpir'] = pynlpir_results_sentence
sentence_df['ltp'] = ltp_results_sentence

names_df.to_excel('names_tokenised.xlsx', "names")
sentence_df.to_excel('sentences_tokenised.xlsx', "sentences")
