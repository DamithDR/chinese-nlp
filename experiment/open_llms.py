import argparse
import sys

import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from experiment.metaphor import load_data


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
        max_length=200,
        do_sample=True,
        max_new_tokens=100,
        top_k=20,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0.2, 'do_sample': True})
    template = """
        隐喻是一种创造性描述方式，通过使用另一事物来描述某一事物，暗示两者在某方面具有相同的特征，同时避免使用“像”或“似”这样的比较词。
        因此，隐喻植物名就是指该植物名中至少有一个词包含隐喻，通过比较或暗示植物与所比较概念或物体的相似之处，唤起特定的形象或情感。
        如：植物名“玉蜀黍”中，“玉”并不是植物本身的特性，而是植物的某一特征和玉的某一特征类似，故而用“玉”来暗指植物像玉一样。因此这个植物名可判定为隐喻植物名。
        你的任务是在给定的中文文本中识别出隐喻植物名。
        你必须确定是否有隐喻植物名，并以json格式回应是或否（取决于是否有隐喻植物名）并回应一个包含名字列表的json对象。
        json对象示例 : {metaphoric_names_found : 'yes',metaphoric_names = ['name1','name2']}
        不需要提供任何其他解释，只需要返回带结果的json对象。

        句子 : {sentence}"""

    prompt = PromptTemplate(template=template, input_variables=["sentence"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()
    out_list = []
    with tqdm(total=len(raw_sentences)) as pbar:
        for sent in raw_sentences:
            response = llm_chain.invoke(sent)
            out_list.extend(response)
            pbar.update(1)

    alias = str(args.model_name).replace('/', '_')
    with open(f'raw_results_from_model{alias}.txt', 'w') as f:
        f.writelines("\n".join(out_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--batch_size', type=int, required=False, default=2, help='batch_size')

    args = parser.parse_args()
    run(args)
