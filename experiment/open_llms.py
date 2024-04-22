import argparse
import sys

import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from experiment.metaphor import load_data


def get_template(language):
    if language == 'zh':
        template = """
            隐喻是一种创造性描述方式，通过使用另一事物来描述某一事物，暗示两者在某方面具有相同的特征，同时避免使用“像”或“似”这样的比较词。
            因此，隐喻植物名就是指该植物名中至少有一个词包含隐喻，通过比较或暗示植物与所比较概念或物体的相似之处，唤起特定的形象或情感。
            如：植物名“玉蜀黍”中，“玉”并不是植物本身的特性，而是植物的某一特征和玉的某一特征类似，故而用“玉”来暗指植物像玉一样。因此这个植物名可判定为隐喻植物名。
            你的任务是在给定的中文文本中识别出隐喻植物名。
            你必须确定是否有隐喻植物名，并以json格式回应是或否（取决于是否有隐喻植物名）并回应一个包含名字列表的json对象。
            json对象示例 : {"metaphoric_names_found" : "yes","metaphoric_names" = ["name1","name2"]}
            不做解释。不承认。只返回带有结果的 json 对象。除了 json 对象之外，不要提供任何其他句子。
    
            句子 : """
        return template
    elif language == 'en':
        template = """
                A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
                Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
                In other words, metaphoric flower name is one that uses a word or phrase that evokes certain qualities, characteristics, or associations beyond the literal description of the flower itself.
                For example, the flower name "Forget-me-not" is metaphorical. While it directly refers to the tiny, delicate blue flowers of the Myosotis genus, its name carries emotional connotations beyond its physical appearance. "Forget-me-not" suggests remembrance, loyalty, and enduring love. It metaphorically implies that the giver of the flower is asking the recipient not to forget them and to remember the bond they share, making it a poignant and symbolic name for this charming flower.
                Your task is to identify metaphorical flower names in the given English text.
                You must identify if there are metaphoric flower names respond as a json format of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
                Example json object : {"metaphoric_names_found" : 'yes',"metaphoric_names" = ['name1','name2']}
                Do not give explanations. Do not acknowledge. Only return json object with the results. Do not provide any other sentence but the json object.
                
                Sentence : """
        return template
    elif language == 'zh_en':
        template = """
                A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
                Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
                In other words, metaphoric flower name is one that uses a word or phrase that evokes certain qualities, characteristics, or associations beyond the literal description of the flower itself.
                For example, the flower name "Forget-me-not" is metaphorical. While it directly refers to the tiny, delicate blue flowers of the Myosotis genus, its name carries emotional connotations beyond its physical appearance. "Forget-me-not" suggests remembrance, loyalty, and enduring love. It metaphorically implies that the giver of the flower is asking the recipient not to forget them and to remember the bond they share, making it a poignant and symbolic name for this charming flower.
                Your task is to identify metaphorical flower names in the given Chinese text.
                You must identify if there are metaphoric flower names respond as a json format of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
                Example json object : {"metaphoric_names_found" : 'yes',"metaphoric_names" = ['name1','name2']}
                Do not give explanations. Do not acknowledge. Only return json object with the results. Do not provide any other sentence but the json object.

                Sentence : """
        return template
    elif language == 'es':
        template = """
                Una metáfora es una forma imaginativa de describir algo refiriéndose a otra cosa que es igual de una manera particular sin usar la palabra "como" o "como".
                Por lo tanto, un nombre de planta metafórico sería un nombre dado a una planta que contiene al menos una palabra metafórica que normalmente establecería una comparación o implicaría una similitud entre la planta y el concepto u objeto con el que se compara, a menudo para evocar un significado particular. imagen o emoción.
                En otras palabras, el nombre de una flor metafórica es aquel que utiliza una palabra o frase que evoca ciertas cualidades, características o asociaciones más allá de la descripción literal de la flor misma.
                Por ejemplo, el nombre de la flor "No me olvides" es metafórico. Si bien se refiere directamente a las diminutas y delicadas flores azules del género Myosotis, su nombre tiene connotaciones emocionales más allá de su apariencia física. "No me olvides" sugiere recuerdo, lealtad y amor duradero. Metafóricamente implica que el donante de la flor le pide al destinatario que no lo olvide y que recuerde el vínculo que comparten, lo que lo convierte en un nombre conmovedor y simbólico para esta encantadora flor.
                Su tarea es identificar nombres de flores metafóricos en el texto en inglés dado.
                Debes identificar si hay nombres de flores metafóricos que responden en formato json de sí/no (dependiendo de si hay un nombre de flor metafórico o no) y la lista de nombres en un objeto json.
                Objeto json de ejemplo: {"metaphoric_names_found": 'yes',"metaphoric_names" = ['nombre1','nombre2']}
                No des explicaciones. No lo reconozcas. Solo devuelve el objeto json con los resultados. No proporcione ninguna otra oración que no sea el objeto json.

                Oración: """
        return template

    elif language == 'es_en':
        template = """
                A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
                Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
                In other words, metaphoric flower name is one that uses a word or phrase that evokes certain qualities, characteristics, or associations beyond the literal description of the flower itself.
                For example, the flower name "Forget-me-not" is metaphorical. While it directly refers to the tiny, delicate blue flowers of the Myosotis genus, its name carries emotional connotations beyond its physical appearance. "Forget-me-not" suggests remembrance, loyalty, and enduring love. It metaphorically implies that the giver of the flower is asking the recipient not to forget them and to remember the bond they share, making it a poignant and symbolic name for this charming flower.
                Your task is to identify metaphorical flower names in the given Spanish text.
                You must identify if there are metaphoric flower names respond as a json format of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
                Example json object : {"metaphoric_names_found" : 'yes',"metaphoric_names" = ['name1','name2']}
                Do not give explanations. Do not acknowledge. Only return json object with the results. Do not provide any other sentence but the json object.
    
                Sentence : """
        return template


def run(args):
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pipeline = pipeline(
        "text-generation",  # task
        model=args.model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        do_sample=True,
        max_new_tokens=100,
        top_k=20,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline,
                              model_kwargs={'temperature': 0.2, 'do_sample': True, 'trust_remote_code': True})

    template = get_template(args.language)

    prompt = PromptTemplate(template=template, input_variables=["sentence"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

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

    raw_sentences = raw_sentences[:2]
    print(raw_sentences)
    print(prompt)
    out_list = []
    with tqdm(total=len(raw_sentences)) as pbar:
        for sent in raw_sentences:
            response = llm_chain.invoke(sent)
            out_list.extend(response)
            pbar.update(1)

    alias = str(args.model_name).replace('/', '_')
    with open(f'raw_results_model_{alias}_{args.language}.txt', 'w') as f:
        f.writelines("\n".join(out_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--batch_size', type=int, required=False, default=2, help='batch_size')
    parser.add_argument('--language', type=str, required=True, default='zh', help='language to run experiment')

    args = parser.parse_args()
    run(args)
