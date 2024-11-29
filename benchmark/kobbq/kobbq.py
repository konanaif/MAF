import os, sys, re
import csv
import json
import torch
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

#
import MAF.benchmark.kobbq.util as util
import MAF.benchmark.kobbq.process_data as pcd
import MAF.benchmark.kobbq.evaluation as eval
from MAF.utils.common import fix_seed

parent_dir = os.environ["PYTHONPATH"]
data_dir = parent_dir + "/MAF/data/kobbq/"


class PreprocessArguments:
    def __init__(self, prompt_id: int):
        self.prompt_id = prompt_id
        self.samples_tsv_path = data_dir + "kobbq_data/KoBBQ_test_samples.tsv"
        self.prompt_tsv_path = data_dir + "0_evaluation_prompts.tsv"
        evaluation_dir = data_dir + "kobbq_result/KoBBQ_test/KoBBQ_test_evaluation"
        self.evaluation_tsv_path = evaluation_dir + f"_{prompt_id}.tsv"
        self.evaluation_json_path = evaluation_dir + f"_{prompt_id}.json"


class InferenceArguments:
    def __init__(
        self,
        prompt_id: int,
        model_name: str,
        custom_model_path=None,
        custom_model_tokenizer=None,
    ):
        self.max_tokens = 30
        self.batch_size = 1
        self.data_path = (
            data_dir + f"kobbq_result/KoBBQ_test/KoBBQ_test_evaluation_{prompt_id}.json"
        )
        self.output_dir = data_dir + f"kobbq_result/outputs/raw/KoBBQ_test_{prompt_id}"
        self.model_name = model_name.replace("/", "-")
        self.is_custom_model = False
        if (custom_model_path != None) and (custom_model_tokenizer != None):
            self.is_custom_model = True
            self.custom_model_path = custom_model_path
            self.custom_model_tokenizer = custom_model_tokenizer


class PostprocessArguments:
    def __init__(self, prompt_id: int, model_name: str):
        self.ooc_path = None
        self.model_name = model_name.replace("/", "-")
        self.prompt_id = prompt_id
        self.predictions_tsv_path = (
            data_dir
            + f"kobbq_result/outputs/raw/KoBBQ_test_{prompt_id}/KoBBQ_test_evaluation_{prompt_id}_{self.model_name}_predictions.tsv"
        )
        self.preprocessed_tsv_path = (
            data_dir + f"kobbq_result/KoBBQ_test/KoBBQ_test_evaluation_{prompt_id}.tsv"
        )
        output_model_dir = (
            data_dir + f"kobbq_result/outputs/processed/KoBBQ_test_{prompt_id}"
        )
        self.output_path = (
            output_model_dir
            + f"/KoBBQ_test_evaluation_{prompt_id}_{self.model_name}.tsv"
        )


class EvaluationArguments:
    def __init__(self, prompt_id: int, model_name: str, test_or_all: str):
        self.topic = "KoBBQ_test_evaluation"
        self.test_or_all = test_or_all
        self.prompt_tsv_path = data_dir + "0_evaluation_prompts.tsv"
        self.prompt_id = prompt_id
        self.model_name = model_name.replace("/", "-")
        self.model_result_tsv_dir = (
            data_dir + f"kobbq_result/outputs/processed/KoBBQ_test_{prompt_id}"
        )
        self.evaluation_result_path = (
            data_dir + f"kobbq_evaluation_result/KoBBQ_test_{prompt_id}.tsv"
        )


def create_prompts(prompt_id: int = 1):
    args = PreprocessArguments(prompt_id)
    pcd.preprocess(args)


def inference(args):
    if not args.data_path.endswith(".json"):
        raise ValueError

    data_path = Path(args.data_path)
    topic = data_path.name.replace(".json", "")
    print(topic)

    model_name = args.model_name

    if args.batch_size != 1 and model_name not in [
        "clova-x",
        "KoAlpaca-Polyglot-12.8B",
    ]:
        raise NotImplementedError

    koalpaca = None
    if model_name in util.KOALPACA_MODEL:  # run with GPU
        koalpaca = util.load_koalpaca(model_name)

    if args.is_custom_model:
        custom_model = util.load_custom_model(
            args.custom_model_path, args.custom_model_tokenizer
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(data_path, "r", encoding="utf-8"))
    prefix = data["prefix"]

    output_path = output_dir / f"{topic}_{model_name}_predictions.tsv"
    if output_path.is_file():
        print(f"Continue on {output_path}")
        done_ids = pd.read_csv(
            output_dir / f"{topic}_{model_name}_predictions.tsv", sep="\t"
        )["guid"].to_list()
    else:
        done_ids = []
        with open(
            output_dir / f"{topic}_{model_name}_predictions.tsv", "w", encoding="utf-8"
        ) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["time", "topic", "guid", "truth", "raw"])

    for i in tqdm(range(0, len(data["data"]), args.batch_size), desc=model_name):
        # instance: prompt, A, B, C, truth, guid
        instances = [
            data["data"][j]
            for j in range(i, min(i + args.batch_size, len(data["data"])))
            if data["data"][j][-1] not in done_ids
        ]

        if not instances:
            continue

        prompt = [prefix + instance[0] for instance in instances]

        if model_name in util.GPT_MODEL:
            result = [
                util.get_gpt_response(
                    prompt[0], model_name, max_tokens=args.max_tokens, greedy=True
                )
            ]
        elif model_name in util.HYPERCLOVA_MODEL:
            result = util.get_hyperclova_response(
                prompt, model_name, max_tokens=args.max_tokens, greedy=True
            )
        elif model_name in util.CLAUDE_MODEL:
            result = [
                util.get_claude_response(
                    prompt[0], model_name, max_tokens=args.max_tokens
                )
            ]
        elif model_name in util.KOALPACA_MODEL:
            result = util.get_koalpaca_response(
                prompt,
                model_name,
                koalpaca,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
        # elif CUSTOM_MODEL in model_name:
        else:
            result = util.get_custom_model_response(
                prompt,
                model_name,
                custom_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
        # else:
        #     raise ValueError(model_name)

        for i, instance in enumerate(instances):
            open_trial = 0
            while True:
                if open_trial >= 10:
                    raise Exception("File Open Fail")

                try:
                    with open(
                        output_dir / f"{topic}_{model_name}_predictions.tsv",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow(
                            [
                                datetime.now(),
                                topic,
                                instance[-1],
                                instance[-2],
                                result[i],
                            ]
                        )
                    break
                except KeyboardInterrupt:
                    raise Exception("Keyboard Interrupt")
                except:
                    print("open failed")
                    open_trial += 1
                    continue

    print(f"{topic} - {model_name} done")


"""
  # model-name = [
  # 'gpt-3.5-turbo',
  # 'gpt-4',
  # 'claude-instant-1.2',
  # 'claude-2.0',
  # 'clova-x',
  # 'KoAlpaca-Polyglot-12.8B']
"""


def run(model_name: str, custom_model_path=None, custom_model_tokenizer=None):
    print(
        f""" model_name {model_name}
             custom_model_path {custom_model_path}
             custom_model_tokenizer {custom_model_tokenizer}"""
    )

    for pmt_id in range(1, 2):  # 6):
        print(f"*********Prompt ID {pmt_id}*********")
        create_prompts(prompt_id=pmt_id)

        infargs = InferenceArguments(
            prompt_id=pmt_id,
            model_name=model_name,
            custom_model_path=custom_model_path,
            custom_model_tokenizer=custom_model_path,
        )
        inference(infargs)
        postprocess_args = PostprocessArguments(prompt_id=pmt_id, model_name=model_name)
        pcd.postprocess_predictions(postprocess_args)
        pcd.predictions_to_evaluation(postprocess_args)

        evaluate_args = EvaluationArguments(
            prompt_id=pmt_id, model_name=model_name, test_or_all="test"
        )
        eval_res = eval.evaluation(evaluate_args)


if __name__ == "__main__":
    run(
        model_name="google-t5/t5-small",
        custom_model_path="google-t5/t5-small",
        custom_model_tokenizer="google-t5/t5-small",
    )
