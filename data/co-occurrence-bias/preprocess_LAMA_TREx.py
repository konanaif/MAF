from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

if __name__ == "__main__":
    """
    'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b', 'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
    'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B-Instruct'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
            "roberta-large",
        ],
    )
    args = parser.parse_args()

    stopword_list = stopwords.words("english")
    capitalized_stopword_list = []
    for word in stopword_list:
        capitalized_stopword_list.append(word.capitalize())
    stopword_list = stopword_list + capitalized_stopword_list

    tokenizers = []
    for model_name in args.model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizers.append(tokenizer)

    relations = {}
    with jsonlines.open("original_LAMA/data/relations.jsonl") as fin:
        for rel in fin.iter():
            relations[rel["relation"]] = rel["template"]

    trex_path = "original_LAMA/data/TREx"
    data_paths = os.listdir(trex_path)
    for i in range(len(data_paths)):
        data_paths[i] = os.path.join(trex_path, data_paths[i])

    prompts = []
    is_valids = {}
    for data_path in data_paths:
        with jsonlines.open(data_path) as fin:
            for idx, sample in tqdm(enumerate(fin.iter())):
                uid = sample["uuid"]
                subj = sample["sub_label"].strip()
                obj = sample["obj_label"].strip()
                rel_id = sample["predicate_id"]

                if obj in stopword_list:
                    continue

                if obj not in is_valids:
                    is_valid = True
                    for tokenizer in tokenizers:
                        input_ids = tokenizer.encode(
                            " " + obj, add_special_tokens=False
                        )
                        if len(input_ids) != 1:
                            is_valid = False
                            break
                    is_valids[obj] = is_valid
                else:
                    is_valid = is_valids[obj]

                if not is_valid:
                    continue

                template = relations[rel_id]
                input_prompt = template.replace("[X]", subj).replace("[Y]", "[MASK]")
                input_prompt_truncated = input_prompt.split("[MASK]")[0].strip()
                output = obj

                prompts.append(
                    {
                        "uid": uid,
                        "subj": subj,
                        "rel_id": rel_id,
                        "input": input_prompt,
                        "truncated_input": input_prompt_truncated,
                        "output": output,
                    }
                )

    out_path = "LAMA_TREx"
    os.makedirs(out_path, exist_ok=True)
    print(len(prompts))
    train, test = train_test_split(prompts, train_size=0.7, random_state=0)
    with open(os.path.join(out_path, "train.json"), "w") as fout:
        json.dump(train, fout)
    with open(os.path.join(out_path, "test.json"), "w") as fout:
        json.dump(test, fout)
    with open(os.path.join(out_path, "all.json"), "w") as fout:
        json.dump(prompts, fout)
