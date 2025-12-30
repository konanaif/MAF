import re
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import defaultdict, OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    group_lang = parser.add_mutually_exclusive_group(required=True)
    group_lang.add_argument("--ko", action="store_true")
    group_lang.add_argument("--en", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--prompt-unk-path", type=str, default="./utils/prompt_unk.csv")
    parser.add_argument("--prompt-unk-id", type=str)
    parser.add_argument("--output-dir", default="../data/")
    parser.add_argument("--gen-context", action="store_true")
    args = parser.parse_args()

    if args.prompt_unk_id is None:
        args.prompt_unk_id = "Ko-42" if args.ko else "En-42"

    return args


def remove_hex(string):
    string = string.strip('\x08') if isinstance(string, str) else string
    return string


def kobbq_test_comb_idx():
    df_kobbq_test_samples = pd.read_csv(
        "https://raw.githubusercontent.com/naver-ai/KoBBQ/main/data/KoBBQ_test_samples.tsv", sep='\t'
    )
    test_comb_idx = pd.DataFrame({
        "template_id": df_kobbq_test_samples["sample_id"].apply(lambda x: '-'.join(x.split('-')[0:2])),
        "comb_idx": df_kobbq_test_samples["sample_id"].apply(lambda x: x.split('-')[2])
    })
    return test_comb_idx.drop_duplicates()


def swap_variants(ordered_dict, k1, k2):
    if k1 not in ordered_dict or k2 not in ordered_dict:
        return ordered_dict
        
    keys = list(ordered_dict.keys())
    values = list(ordered_dict.values())

    index1 = keys.index(k1)
    index2 = keys.index(k2)

    keys[index1], keys[index2] = keys[index2], keys[index1]
    values[index1], values[index2] = values[index2], values[index1]

    swapped_dict = OrderedDict(zip(keys, values))
    return swapped_dict


def substitute(sentence, comb_idx_dict):
    pattern = re.compile(r'(?P<s>\[(?P<k>\w+):(?P<v>[^\]]+)\])') 

    def replace(match):
        match_dict = match.groupdict()
        k, v = match_dict['k'].strip(), match_dict['v']
        attr = [w.strip() for w in v.split(',')][comb_idx_dict[k]]
        return attr

    return re.sub(pattern, replace, sentence)


def substitute_one_other(dis_context, comb_idx_dict, n1_obfdis_adj=None, n2_obfdis_adj=None, n1_obfdis_post=None, n2_obfdis_post=None):
    pattern = re.compile(r'(?P<s>([Tt]he )?\[(?P<k>\w+):(?P<v>[^\]]+)\])')
    
    if args.ko:
        if n1_obfdis_adj and n1_obfdis_post or n2_obfdis_adj and n2_obfdis_post:
            raise ValueError

        pron_dict = {
            "one": {
                "N1": "한" if n1_obfdis_adj else "한 사람",
                "N2": "한" if n2_obfdis_adj else "한 사람"
            },
            "other": {
                "N1": "다른 한" if n1_obfdis_adj else "다른 한 사람",
                "N2": "다른 한" if n2_obfdis_adj else "다른 한 사람"
            }
        }

        posts = {}
        if n1_obfdis_post:
            posts["N1"] = n1_obfdis_post.split(", ")
        if n2_obfdis_post:
            posts["N2"] = n2_obfdis_post.split(", ")
    
    else:
        pron_dict = {
            "one": {
                "N1": "one",
                "N2": "one"
            },
            "other": {
                "N1": "the other",
                "N2": "the other"
            }
        }
        posts = {}

    n = {"N1": 0, "N2": 0}
    dis2obf = {} # N1: one
    obf2dis = defaultdict(dict) # one: {one+post: attr}

    def get_post(k):
        if k in posts:
            if n[k] < len(posts[k]):
                post = posts[k][n[k]]
            else:
                post = posts[k][0]
        else:
            post = ""
        return post
    
    def replace(match):
        match_dict = match.groupdict()
        k, v = match_dict['k'].strip(), match_dict['v']
        attr = [w.strip() for w in v.split(',')][comb_idx_dict[k]]
        if k in n:
            if "one" not in obf2dis:
                one = pron_dict["one"][k]
                post = get_post(k)
                obf2dis["one"][one + post] = attr
                dis2obf[k] = "one"
                n[k] += 1
                return one + post
            elif k not in dis2obf:
                other = pron_dict["other"][k]
                post = get_post(k)
                obf2dis["other"][other + post] = attr
                dis2obf[k] = "other"
                n[k] += 1
                return other + post
            else:
                pronoun = pron_dict[dis2obf[k]][k]
                post = get_post(k)
                obf2dis[dis2obf[k]][pronoun + post] = attr
                n[k] += 1
                return pronoun + post
        else:
            return attr
    
    obfdis_input, obfdis_output = "", dis_context
    while obfdis_input != obfdis_output:
        obfdis_input = obfdis_output
        obfdis_output = re.sub(pattern, replace, obfdis_input, 1)

    if args.en:
        cap = re.compile(r'(^\w+)|(?<=[.!?])\s+(\w+)')
        obfdis_output = cap.sub(lambda x: x.group().capitalize(), obfdis_output)
    
    return obfdis_output, dict(obf2dis)


def bbq2bbg(df_bbq, unknown):
    df_bbq = df_bbq.apply(remove_hex)

    id2comb_idx = {}
    bbg = OrderedDict()
    bbg["id"] = []
    bbg["category"] = []
    bbg["template_id"] = []
    if args.gen_context:
        bbg["context"] = []
    bbg["amb_context"] = []
    bbg["dis_context"] = []
    bbg["obfdis_context"] = []
    bbg["obf2dis"] = []
    bbg["qtype"] = []
    bbg["question"] = []
    bbg["choices"] = []
    bbg["dis_answer"] = []
    bbg["biased_answer"] = []
    bbg["stereotype"] = []
    bbg["target_group"] = []
    bbg["n1_info"] = []
    bbg["n2_info"] = []
    
    if args.ko and args.random_seed == 42:
        test_comb_idx = kobbq_test_comb_idx()
    
    for _, row in tqdm(df_bbq.iterrows(), total=len(df_bbq)):
        
        category = row["Category"].lower()
        template_idx = row["Template_ID"]
        context_ab = row["Version"]
        
        names = row["Names"]
        lexicon = row["Lexical_diversity"]

        comb_dict = OrderedDict()
        for variants in [names, lexicon]:
            if not variants:
                continue
            for kv in variants.split(';'):
                k = kv.split(':')[0].strip()
                vs = [v.strip() for v in kv.split(':')[1].replace('[', '').replace(']', '').split(',')]
                comb_dict[k] = vs

        if context_ab in ['b', 'd']:
            comb_dict = swap_variants(comb_dict, 'N1', 'N2')
            comb_dict = swap_variants(comb_dict, 'X1', 'X2')

        comb_list = list(product(*comb_dict.values()))
        if args.all:
            sample_idxs = range(1, len(comb_list) + 1)
        elif args.ko and args.random_seed == 42:
            try:
                comb_idx = int(test_comb_idx[test_comb_idx["template_id"] == f"{category}-{template_idx:03d}{context_ab}"]["comb_idx"].item())
            except:
                # print(f"{category}-{template_idx:03d}{context_ab}")
                # print(test_comb_idx[test_comb_idx["template_id"] == f"{category}-{template_idx:03d}{context_ab}"])
                pass
            sample_idxs = [comb_idx]
        else:
            if context_ab == 'b':
                comb_idx = id2comb_idx[f"{category}-{template_idx:03d}-a"]
            elif context_ab == 'd':
                comb_idx = id2comb_idx[f"{category}-{template_idx:03d}-c"]
            else:
                comb_idx = random.randint(1, len(comb_list))
            sample_idxs = [comb_idx]

        def make_sample(sample_idx):
            template_id = f"{category}-{template_idx:03d}-{context_ab}"
            id2comb_idx[template_id] = sample_idx
            sample_id = f"{template_id}-{sample_idx:03d}"
            comb = comb_list[sample_idx - 1]
        
            comb_idx_dict = {x_item[0]: x_item[1].index(x) for x, x_item in zip(comb, comb_dict.items())}
        
            amb_context_temp = row["Ambiguous_context"]
            dis_context_temp = row["Disambiguating_context"]
            amb_context = substitute(amb_context_temp, comb_idx_dict)
            dis_context = substitute(dis_context_temp, comb_idx_dict)
            if args.ko:
                n1_obfdis_adj, n2_obfdis_adj = row["N1_obfdis_adj"], row["N2_obfdis_adj"]
                n1_obfdis_post, n2_obfdis_post = row["N1_obfdis_post"], row["N2_obfdis_post"]
                obfdis_context, obf2dis = substitute_one_other(dis_context_temp, comb_idx_dict, n1_obfdis_adj, n2_obfdis_adj, n1_obfdis_post, n2_obfdis_post)
            else:
                obfdis_context, obf2dis = substitute_one_other(dis_context_temp, comb_idx_dict)
        
            biased_question_ans = substitute(row["Biased_question_answer"], comb_idx_dict)
            cnt_biased_question_ans = substitute(row["Counter-biased_question_answer"], comb_idx_dict)
        
            for _ in range(2):
                bbg["template_id"].append(template_id)
                if args.gen_context:
                    bbg["context"].append(amb_context.strip() + ' ' + obfdis_context.strip())
                bbg["amb_context"].append(amb_context)
                bbg["dis_context"].append(dis_context)
                bbg["obfdis_context"].append(obfdis_context)
                bbg["obf2dis"].append(obf2dis)
                bbg["choices"].append([biased_question_ans, cnt_biased_question_ans, unknown])
                bbg["category"].append(category)
                bbg["stereotype"].append(row["Stereotype"])
                bbg["target_group"].append(row["Target_group"])
                bbg["n1_info"].append(row["N1_info"])
                bbg["n2_info"].append(row["N2_info"])
        
            # biased question
            bbg["id"].append(sample_id + "-bsd")
            bbg["qtype"].append("bsd")
            bbg["question"].append(row["Biased_question"])
            bbg["dis_answer"].append(biased_question_ans)
            if context_ab in ['a', 'c']:
                bbg["biased_answer"].append(cnt_biased_question_ans)
            elif context_ab in ['b', 'd']:
                bbg["biased_answer"].append(biased_question_ans)
            else:
                raise ValueError
        
            # counter-biased question
            bbg["id"].append(sample_id + "-cnt")
            bbg["qtype"].append("cnt")
            bbg["question"].append(row["Counter-biased_question"])
            bbg["dis_answer"].append(cnt_biased_question_ans)
            if context_ab in ['a', 'c']:
                bbg["biased_answer"].append(biased_question_ans)
            elif context_ab in ['b', 'd']:
                bbg["biased_answer"].append(cnt_biased_question_ans)
            else:
                raise ValueError

        for sample_idx in sample_idxs:
            make_sample(sample_idx)

    df = pd.DataFrame(bbg)
    # df["choices"] = df["choices"].apply(lambda x: list(np.random.permutation(x)))
    df["choices"] = df["choices"].apply(lambda x: list(map(str, np.random.permutation(x))))

    return df


if __name__ == '__main__':
    args = parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.ko:
        language = "Ko"
    else: # args.en
        language = "En"

    print("BBQ to BBG...")
    template_path = f"../data/{language}BBG_templates.csv"
    df_input = pd.read_csv(template_path)
    df_input = df_input.where(pd.notna(df_input), None)

    df_unk = pd.read_csv(args.prompt_unk_path)
    unk = df_unk[df_unk["unk_id"] == args.prompt_unk_id]["unknown"].item()

    df_output = bbq2bbg(df_input, unk)

    print("Saving...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        output_path = output_dir / f"{language}BBG_{args.prompt_unk_id}_{args.random_seed}_all.csv"
    else:
        output_path = output_dir / f"{language}BBG_{args.prompt_unk_id}_{args.random_seed}_eval.csv"
    df_output.to_csv(output_path, index=False)
