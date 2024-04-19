# Load model directly
import argparse
import gc

import jieba
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from experiment.metaphor import load_data

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    initial_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = nn.DataParallel(initial_model).to(device)

    train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()

    # testing
    # dataset = dataset[100:102]
    # prompt_list = generate_prompts(dataset)
    # prompt_list = [f'<s>[INST] {prompt} [/INST]'for prompt in prompt_list]

    generation_config = GenerationConfig(
        max_new_tokens=100, do_sample=True, top_k=20, eos_token_id=initial_model.config.eos_token_id,
        pad_token_id=initial_model.config.eos_token_id, temperature=0.2,
        num_return_sequences=1)

    prompt = """
        隐喻是一种创造性描述方式，通过使用另一事物来描述某一事物，暗示两者在某方面具有相同的特征，同时避免使用“像”或“似”这样的比较词。
        因此，隐喻植物名就是指该植物名中至少有一个词包含隐喻，通过比较或暗示植物与所比较概念或物体的相似之处，唤起特定的形象或情感。
        如：植物名“玉蜀黍”中，“玉”并不是植物本身的特性，而是植物的某一特征和玉的某一特征类似，故而用“玉”来暗指植物像玉一样。因此这个植物名可判定为隐喻植物名。
        你的任务是在给定的中文文本中识别出隐喻植物名。
        你必须确定是否有隐喻植物名，并以json格式回应是或否（取决于是否有隐喻植物名）并回应一个包含名字列表的json对象。
        json对象示例 : {metaphoric_names_found : 'yes',metaphoric_names = ['name1','name2']}
        不需要提供任何其他解释，只需要返回带结果的json对象。

        句子 : """

    outputs = []
    total_sent = len(raw_sentences)

    current_number = 0
    out_list = []
    for i in tqdm(range(args.batch_size, total_sent, args.batch_size)):
        if current_number + i > total_sent:
            sents = raw_sentences[current_number:]
        else:
            sents = raw_sentences[current_number:current_number + i]

        data_batch = [prompt + s for s in sents]
        current_number += i
        encoding = tokenizer(data_batch, padding=True, truncation=False, return_tensors="pt").to(model.device)
        # do the inference
        outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                 generation_config=generation_config)
        detach = outputs.detach().cpu().numpy()
        outputs = detach.tolist()
        out_list.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])
        clear_gpu_memory()

    # for sentence, gold_tags in zip(raw_sentences, gold_tags):
    #     prompt += sentence
    #
    #     encoding = tokenizer(prompt, padding=True, truncation=False, return_tensors="pt").to(model.device)
    #     outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
    #                              generation_config=generation_config)
    #     detach = outputs.detach().cpu().numpy()
    #     outputs = detach.tolist()
    #
    #     out_list.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])

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
