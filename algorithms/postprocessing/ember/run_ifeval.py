import json
import re
import numpy as np
import argparse
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import csv

from MAF.algorithms.postprocessing.ember.gen_util import (
    greedy_decoding_mistral,
    greedy_decoding_llama,
    gpt4_answer,
)
from MAF.algorithms.postprocessing.ember.if_util import (
    prepare_if_inputs_rule,
    prepare_if_inputs_gpt,
    output_label_detector,
    prompt_generation,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--jobid")
    parser.add_argument("--eval_model")
    parser.add_argument("--eval_only")
    parser.add_argument("--max_token", type=int, default=20)
    args = parser.parse_args()
    return args


def ember_if_main(args):
    print(args)
    random.seed(1004)
    with open(args.data_dir) as f:
        data = json.load(f)
    # em_pairs=['pw', 'sw', 'ps', 'ss','pp','ww', 'sp', 'ws', 'wp']
    # em_pairs=['pp']
    # em_pairs=['sw', 'pp', 'ws']
    em_pairs = ["pp", "wp"]
    # em_pairs=['pp',  'ps']
    if args.eval_only.lower() == "false":
        if "llama" in args.eval_model.lower():
            pipeline = transformers.pipeline(
                "text-generation",
                model="meta-llama/" + args.eval_model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        em_pair_inputs = defaultdict(list)
        for em_pair in em_pairs:
            for d in data:
                out_1_s = d["output_1_str"]
                out_1_p = d["output_1"]
                out_1_w = d["output_1_weak"]

                out_2_s = d["output_2_str"]
                out_2_p = d["output_2"]
                out_2_w = d["output_2_weak"]

                if em_pair[0] == "s":
                    if em_pair[1] == "s":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_s, out_2_s
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_s, out_1_s
                        )
                    elif em_pair[1] == "w":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_s, out_2_w
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_w, out_1_s
                        )
                    elif em_pair[1] == "p":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_s, out_2_p
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_p, out_1_s
                        )
                    else:
                        raise TypeError

                elif em_pair[0] == "w":
                    if em_pair[1] == "s":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_w, out_2_s
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_s, out_1_w
                        )

                    elif em_pair[1] == "w":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_w, out_2_w
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_w, out_1_w
                        )

                    elif em_pair[1] == "p":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_w, out_2_p
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_p, out_1_w
                        )

                    else:
                        raise TypeError

                elif em_pair[0] == "p":
                    if em_pair[1] == "s":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_p, out_2_s
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_s, out_1_p
                        )

                    elif em_pair[1] == "w":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_p, out_2_w
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_w, out_1_p
                        )

                    elif em_pair[1] == "p":
                        d["{}_input".format(em_pair)] = prompt_generation(
                            d["input"], out_1_p, out_2_p
                        )
                        d["{}_input_rev".format(em_pair)] = prompt_generation(
                            d["input"], out_2_p, out_1_p
                        )

                    else:
                        raise TypeError
                else:
                    raise TypeError

                em_pair_inputs[em_pair].append(d["{}_input".format(em_pair)])
                em_pair_inputs["{}_rev".format(em_pair)].append(
                    d["{}_input_rev".format(em_pair)]
                )

        em_pair_outputs = {}

        for i in range(2):
            if "gpt" in args.eval_model:
                if "mini" in args.eval_model:
                    for em_pair in em_pairs:
                        em_pair_outputs[em_pair] = gpt4_answer(
                            em_pair_inputs[em_pair], "gpt-4o-mini", args.max_token
                        )
                        em_pair_outputs["{}_rev".format(em_pair)] = gpt4_answer(
                            em_pair_inputs["{}_rev".format(em_pair)],
                            "gpt-4o-mini",
                            args.max_token,
                        )
                elif "4o" in args.eval_model:
                    for em_pair in em_pairs:
                        em_pair_outputs[em_pair] = gpt4_answer(
                            em_pair_inputs[em_pair], "gpt-4o", args.max_token
                        )
                        em_pair_outputs["{}_rev".format(em_pair)] = gpt4_answer(
                            em_pair_inputs["{}_rev".format(em_pair)],
                            "gpt-4o",
                            args.max_token,
                        )
                elif "35" in args.eval_model:
                    for em_pair in em_pairs:
                        em_pair_outputs[em_pair] = gpt4_answer(
                            em_pair_inputs[em_pair], "gpt-3.5-turbo", args.max_token
                        )
                        em_pair_outputs["{}_rev".format(em_pair)] = gpt4_answer(
                            em_pair_inputs["{}_rev".format(em_pair)],
                            "gpt-3.5-turbo",
                            args.max_token,
                        )
                elif "4-turbo" in args.eval_model:
                    for em_pair in em_pairs:
                        em_pair_outputs[em_pair] = gpt4_answer(
                            em_pair_inputs[em_pair], "gpt-4-turbo", args.max_token
                        )
                        em_pair_outputs["{}_rev".format(em_pair)] = gpt4_answer(
                            em_pair_inputs["{}_rev".format(em_pair)],
                            "gpt-4-turbo",
                            args.max_token,
                        )

            elif "llama" in args.eval_model.lower():
                for em_pair in em_pairs:
                    em_pair_outputs[em_pair] = greedy_decoding_llama(
                        pipeline, em_pair_inputs[em_pair], args.max_token
                    )
                    em_pair_outputs["{}_rev".format(em_pair)] = greedy_decoding_llama(
                        pipeline,
                        em_pair_inputs["{}_rev".format(em_pair)],
                        args.max_token,
                    )

            for idx, d in enumerate(data):
                for em_pair in em_pairs:
                    d[
                        "{}_output_{}".format(em_pair, args.eval_model)
                    ] = em_pair_outputs[em_pair][idx]
                    d[
                        "{}_output_rev_{}".format(em_pair, args.eval_model)
                    ] = em_pair_outputs["{}_rev".format(em_pair)][idx]

            directory = os.getcwd()
            if not (Path.cwd() / "out" / "{}".format(args.jobid)).exists():
                os.makedirs(("{}/out/{}".format(directory, args.jobid)), exist_ok=True)

            with open(
                "{}/out/{}/ember_if_{}.json".format(directory, args.jobid, i), "w"
            ) as f:
                json.dump(data, f)

    # Evaluation Start
    csv_header = [["Evaluator: {} result".format(args.eval_model)]]
    first_row = [""]
    for d in em_pairs:
        first_row.append(d)
    csv_header.append(first_row)
    accuracy = defaultdict(list)
    absolute = defaultdict(list)
    relative = defaultdict(list)
    c2i = defaultdict(int)
    i2c = defaultdict(int)
    for idx, d in enumerate(data):
        for em_pair in em_pairs:
            d["{}_result_{}".format(em_pair, args.eval_model)] = np.average(
                [
                    output_label_detector(
                        d["{}_output_{}".format(em_pair, args.eval_model)]
                    )
                    == 1,
                    output_label_detector(
                        d["{}_output_rev_{}".format(em_pair, args.eval_model)]
                    )
                    == 2,
                ]
            )
            accuracy[em_pair].append(d["{}_result_{}".format(em_pair, args.eval_model)])

    for idx, d in enumerate(data):
        for em_pair in em_pairs:
            pp_output = (
                1
                if output_label_detector(d["pp_output_{}".format(args.eval_model)]) == 1
                else 0
            )
            pp_output_rev = (
                1
                if output_label_detector(d["pp_output_rev_{}".format(args.eval_model)])
                == 2
                else 0
            )
            em_output = (
                1
                if output_label_detector(
                    d["{}_output_{}".format(em_pair, args.eval_model)]
                )
                == 1
                else 0
            )
            em_output_rev = (
                1
                if output_label_detector(
                    d["{}_output_rev_{}".format(em_pair, args.eval_model)]
                )
                == 2
                else 0
            )
            absolute[em_pair].append(
                np.average(
                    [abs(pp_output - em_output), abs(pp_output_rev - em_output_rev)]
                )
            )
            relative[em_pair].append(
                np.average([(em_output - pp_output), (em_output_rev - pp_output_rev)])
            )

            if pp_output == 1:
                if em_output == 0:
                    c2i[em_pair] += 1
            else:
                if em_output == 1:
                    i2c[em_pair] += 1

            if pp_output_rev == 1:
                if em_output_rev == 0:
                    c2i[em_pair] += 1
            else:
                if em_output_rev == 1:
                    i2c[em_pair] += 1

    categories = []
    values = []
    for k, v in accuracy.items():
        categories.append(k.upper())
        values.append(round(np.average(v) * 100, 1))
    plt.bar(categories, values)
    plt.title("Evaluator: {}".format(args.eval_model))
    plt.xlabel("Type")
    plt.ylabel("Surface Accuracy")

    for i, value in enumerate(values):
        plt.text(i, value + 0.005, str(value), ha="center", va="bottom")

    positions_to_divide = [len(values) // 3, 2 * len(values) // 3]

    for pos in positions_to_divide:
        plt.axvline(x=pos - 0.5, color="gray", linestyle="--", alpha=0.7)

    a = np.average(accuracy["pp"]) * 100  # Set the value of 'a'
    plt.axhline(y=a, color="red", linestyle="--", linewidth=1.5)
    plt.show()

    directory = os.getcwd()
    if not (Path.cwd() / "out" / "{}".format(args.jobid)).exists():
        os.makedirs(("{}/out/{}".format(directory, args.jobid)), exist_ok=True)
    with open("{}/out/{}/em_bench.json".format(directory, args.jobid), "w") as f:
        json.dump(data, f)
    plt.savefig(
        "{}/out/{}/{}_model_acc.png".format(directory, args.jobid, args.eval_model)
    )

    absolute_result = ["Absolute"]
    relative_result = ["Relative"]
    surface_result = ["Surface Acc"]
    for em_pair in em_pairs:
        surface_result.append(
            "{} / ({})".format(
                round(np.average(accuracy[em_pair]) * 100, 1),
                round(np.average(relative[em_pair]) * 100, 1),
            )
        )
        absolute_result.append(
            "{} ({} / {})".format(
                round(np.average(absolute[em_pair]) * 100, 1),
                round(c2i[em_pair] / 2 / len(data) * 100, 1),
                round(i2c[em_pair] / 2 / len(data) * 100, 1),
            )
        )

    csv_header.append(["Surface Accuracy"])
    csv_header.append(surface_result)
    csv_header.append(["Absolute Accuracy"])
    csv_header.append(absolute_result)

    with open(
        "{}/out/{}/if_eval_{}_result.csv".format(
            directory, args.jobid, args.eval_model
        ),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_header)

    return {
        "em_pairs": em_pairs,
        "surface_acc": surface_result[1:],
        "absolute_acc": absolute_result[1:],
    }


def migitage_ember_if():
    class args:
        def __init__(self):
            self.data_dir = "MAF/data/ember/if/ember_if.json"
            self.max_token = 10
            self.eval_only = "false"
            self.eval_model = "gpt-4-mini"
            self.jobid = 0

    result = ember_if_main(args())
    return result


if __name__ == "__main__":
    result = migitage_ember_if()
    print(result)
