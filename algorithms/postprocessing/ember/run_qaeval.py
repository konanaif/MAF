import json
import re
import numpy as np
import argparse
import random
import os
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
from MAF.algorithms.postprocessing.ember.qa_util import (
    prepare_qa_inputs_rule,
    prepare_qa_inputs_gpt,
    integer_detector,
    yesno_detector,
    prompt_generation,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--data_type")
    parser.add_argument("--jobid")
    parser.add_argument("--eval_model")
    parser.add_argument("--eval_only")
    parser.add_argument("--scoring")
    parser.add_argument("--correctness")
    parser.add_argument("--max_token", type=int, default=20)

    args = parser.parse_args()
    return args


def ember_qa_main(args):
    print(args)
    random.seed(1004)
    if args.scoring.lower() == "likert":
        scoring = True
    elif args.scoring.lower() == "yesno":
        scoring = False
    else:
        raise KeyError
    data = {}
    model_type = ["gpt4", "newbing"]
    total_result = {
        "acc": {
            "gpt4": {
                f"{args.correctness}_samples": 0,
                "suf": {"str": "0", "plain": "0", "weak": "0"},
                "abs": {"str": "0", "plain": "0", "weak": "0"},
            },
            "newbing": {
                f"{args.correctness}_samples": 0,
                "suf": {"str": "0", "plain": "0", "weak": "0"},
                "abs": {"str": "0", "plain": "0", "weak": "0"},
            },
        },
        "matrix": {
            "gpt4": {"str": "0", "plain": "0", "weak": "0"},
            "newbing": {"str": "0", "plain": "0", "weak": "0"},
        },
    }

    em_types = ["str", "plain", "weak"]
    for reader in model_type:
        with open(args.data_dir + "ember_qa_{}.json".format(reader)) as f:
            orig = json.load(f)
        data[reader] = orig
    if args.eval_only.lower() == "false":
        if "llama" in args.eval_model.lower():
            pipeline = transformers.pipeline(
                "text-generation",
                model="meta-llama/" + args.eval_model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        for i in range(2):
            for reader in model_type:
                # Prepare Input for the Evaluations
                for d in data[reader]:
                    for em_type in em_types:
                        d[
                            "answer_{}_{}_input".format(reader, em_type)
                        ] = prompt_generation(
                            scoring=scoring,
                            question=d["question"][0].upper() + d["question"][1:],
                            output=d["answer_{}_{}".format(reader, em_type)],
                            reference=d["golden_answer"][0],
                        )

                em_type_inputs = defaultdict(list)
                for em_type in em_types:
                    for d in data[reader]:
                        em_type_inputs[em_type].append(
                            d["answer_{}_{}_input".format(reader, em_type)]
                        )
                em_type_outputs = {}
                if "gpt" in args.eval_model:
                    if "mini" in args.eval_model:
                        for em_type in em_types:
                            em_type_outputs[em_type] = gpt4_answer(
                                em_type_inputs[em_type], "gpt-4o-mini", args.max_token
                            )

                    elif "4o" in args.eval_model:
                        for em_type in em_types:
                            em_type_outputs[em_type] = gpt4_answer(
                                em_type_inputs[em_type], "gpt-4o", args.max_token
                            )
                    elif "35" in args.eval_model:
                        for em_type in em_types:
                            em_type_outputs[em_type] = gpt4_answer(
                                em_type_inputs[em_type], "gpt-3.5-turbo", args.max_token
                            )
                    elif "4-turbo" in args.eval_model:
                        for em_type in em_types:
                            em_type_outputs[em_type] = gpt4_answer(
                                em_type_inputs[em_type], "gpt-4-turbo", args.max_token
                            )

                elif "llama" in args.eval_model.lower():
                    for em_type in em_types:
                        em_type_outputs[em_type] = greedy_decoding_llama(
                            pipeline, em_type_inputs[em_type], args.max_token
                        )

                for idx, d in enumerate(data[reader]):
                    for em_type in em_types:
                        d[
                            "answer_{}_{}_{}_output".format(
                                reader, em_type, args.eval_model
                            )
                        ] = em_type_outputs[em_type][idx]

            directory = os.getcwd()
            if not (Path.cwd() / "out" / "{}".format(args.jobid)).exists():
                os.makedirs(("{}/out/{}".format(directory, args.jobid)), exist_ok=True)

            for reader in model_type:
                with open(
                    "{}/out/{}/ember_qa_{}.json".format(
                        directory, args.jobid, reader, i
                    ),
                    "w",
                ) as f:
                    json.dump(data[reader], f)

    # Evaluation Start
    csv_header = [["Evaluator: {} result".format(args.eval_model)]]
    csv_header2 = [["Evaluator: {} result".format(args.eval_model)]]

    for reader in model_type:
        if args.correctness == "true":
            data[reader] = [d for d in data[reader] if d["judge_{}".format(reader)]]
        else:
            data[reader] = [d for d in data[reader] if not d["judge_{}".format(reader)]]
        first_row = ["Reader: {}".format(reader)]
        for d in em_types:
            first_row.append(d)
        csv_header.append(["{} {} samples".format(args.correctness, len(data[reader]))])
        total_result["acc"][reader]["true_samples"] = 0
        total_result["acc"][reader]["false_samples"] = 0
        total_result["acc"][reader][f"{args.correctness}_samples"] = len(data[reader])

        csv_header.append(first_row)
        csv_header2.append(["Reader:", reader.upper()])
        accuracy = defaultdict(list)
        absolute = defaultdict(list)
        relative = defaultdict(list)
        c2i = defaultdict(int)
        i2c = defaultdict(int)

        for idx, d in enumerate(data[reader]):
            for em_type in em_types:
                if scoring:
                    d[
                        "answer_{}_{}_{}_result".format(
                            reader, em_type, args.eval_model
                        )
                    ] = integer_detector(
                        d[
                            "answer_{}_{}_{}_output".format(
                                reader, em_type, args.eval_model
                            )
                        ]
                    )
                else:
                    human = 1 if d["judge_{}".format(reader)] else 0
                    if human == yesno_detector(
                        d[
                            "answer_{}_{}_{}_output".format(
                                reader, em_type, args.eval_model
                            )
                        ]
                    ):
                        d[
                            "answer_{}_{}_{}_result".format(
                                reader, em_type, args.eval_model
                            )
                        ] = 1
                    else:
                        d[
                            "answer_{}_{}_{}_result".format(
                                reader, em_type, args.eval_model
                            )
                        ] = 0
                accuracy[em_type].append(
                    d["answer_{}_{}_{}_result".format(reader, em_type, args.eval_model)]
                )

        for idx, d in enumerate(data[reader]):
            for em_type in em_types:
                if scoring:
                    p_output = integer_detector(
                        d["answer_{}_plain_{}_output".format(reader, args.eval_model)]
                    )
                    em_output = integer_detector(
                        d[
                            "answer_{}_{}_{}_output".format(
                                reader, em_type, args.eval_model
                            )
                        ]
                    )

                else:
                    human = 1 if d["judge_{}".format(reader)] else 0
                    p_output = (
                        1
                        if human
                        == yesno_detector(
                            d[
                                "answer_{}_plain_{}_output".format(
                                    reader, args.eval_model
                                )
                            ]
                        )
                        else 0
                    )
                    em_output = (
                        1
                        if human
                        == yesno_detector(
                            d[
                                "answer_{}_{}_{}_output".format(
                                    reader, em_type, args.eval_model
                                )
                            ]
                        )
                        else 0
                    )
                absolute[em_type].append(abs(em_output - p_output))
                relative[em_type].append(em_output - p_output)
                if p_output == 1:
                    if em_output == 0:
                        c2i[em_type] += 1
                else:
                    if em_output == 1:
                        i2c[em_type] += 1

        categories = []
        values = []
        for k, v in accuracy.items():
            none_cnt = v.count(None)
            categories.append(k.upper())
            none_outputs = [d for d in v if d != None]

            if scoring:
                values.append(round(np.average(none_outputs), 1))
            else:
                values.append(round(np.average(none_outputs) * 100, 1))

        plt.bar(categories, values)
        plt.title(
            "Dataset: {} / Reader: {} / Evaluator: {}".format(
                args.data_type, reader, args.eval_model
            )
        )
        plt.xlabel("Type")
        plt.ylabel("Average Accuracy")

        for i, value in enumerate(values):
            plt.text(i, value + 0.005, str(value), ha="center", va="bottom")

        positions_to_divide = [len(values) // 3, 2 * len(values) // 3]
        for pos in positions_to_divide:
            plt.axvline(x=pos - 0.5, color="gray", linestyle="--", alpha=0.7)

        if scoring:
            a = np.average(accuracy["plain"])  # Set the value of 'a'
        else:
            a = np.average(accuracy["plain"]) * 100  # Set the value of 'a'

        plt.axhline(y=a, color="red", linestyle="--", linewidth=1.5)
        plt.show()
        directory = os.getcwd()
        if not (Path.cwd() / "out" / "{}".format(args.jobid)).exists():
            os.makedirs(("{}/out/{}".format(directory, args.jobid)), exist_ok=True)

        plt.savefig(
            "{}/out/{}/{}_{}_model_ac.png".format(
                directory, args.jobid, reader, args.eval_model
            )
        )

        plt.clf()

        surface_result = ["Surface ACC"]
        relative_result = ["Relative"]
        absolute_result = ["Absolute"]
        for em_type in em_types:
            if scoring:
                suf_acc = round(np.average(accuracy[em_type]), 1)
                abs_acc = round(np.average(absolute[em_type]), 1)
            else:
                suf_acc = "{} ({})".format(
                    round(np.average(accuracy[em_type]) * 100, 1),
                    round(np.average(relative[em_type]) * 100, 1),
                )
                abs_acc = "{} ({} / {})".format(
                    round(np.average(absolute[em_type]) * 100, 1),
                    round(c2i[em_type] / len(data[reader]) * 100, 1),
                    round(i2c[em_type] / len(data[reader]) * 100, 1),
                )
            surface_result.append(suf_acc)
            absolute_result.append(abs_acc)
            total_result["acc"][reader]["suf"][em_type] = suf_acc
            total_result["acc"][reader]["abs"][em_type] = abs_acc

        csv_header.append(["Surface Accuracy"])
        csv_header.append(surface_result)
        csv_header.append(["Absolute Result"])
        csv_header.append(absolute_result)

        with open(
            "{}/out/{}/qa_{}_eval_{}_result.csv".format(
                directory, args.jobid, args.data_type, args.eval_model
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerows(csv_header)
        csv_header.append([])

        for em_type in em_types:
            csv_header2.append(["Epistemic Marker: ", em_type])
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            for idx, d in enumerate(data[reader]):
                human_judge = 1 if d["judge_{}".format(reader)] else 0
                model_judge = yesno_detector(
                    d["answer_{}_{}_{}_output".format(reader, em_type, args.eval_model)]
                )
                if human_judge == 1:
                    if model_judge == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if model_judge == 1:
                        fp += 1
                    else:
                        tn += 1
            csv_header2.append(["", "", "Human"])
            csv_header2.append(["", "", "True", "False"])
            csv_header2.append(["Evaluator", "True", tp, fp])
            csv_header2.append(["Evaluator", "False", fn, tn])
            csv_header2.append(
                [
                    "ACC against human:",
                    "{:.2f}".format((tp + tn) / (tp + tn + fp + fn) * 100),
                ]
            )
            csv_header2.append([])
            total_result["matrix"][reader][em_type] = "{:.2f}".format(
                (tp + tn) / (tp + tn + fp + fn) * 100
            )
        with open(
            "{}/out/{}/qa_{}_eval_{}_matrix_result.csv".format(
                directory, args.jobid, args.data_type, args.eval_model
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerows(csv_header2)
        csv_header2.append([])
        csv_header2.append(["-", "-", "-", "-"])

    return total_result


def migitage_ember_qa(eval_model: str = "gpt-4-mini"):
    class args:
        def __init__(self):
            self.data_type = "integ"
            self.data_dir = "MAF/data/ember/qa/"
            self.correctness = "true"
            self.max_token = 10
            self.eval_only = "false"
            self.eval_model = eval_model
            self.scoring = "yesno"
            self.jobid = 0

    result = ember_qa_main(args())
    return result


if __name__ == "__main__":
    result = migitage_ember_qa()
    print(result)
