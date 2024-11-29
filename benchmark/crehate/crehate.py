import os, sys
import re
import csv
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from easydict import EasyDict

from MAF.benchmark.crehate.util import inference, load_model

parent_dir = os.environ["PYTHONPATH"]
data_dir = parent_dir + "/MAF/data/crehate/"


def infer_model_simple(
    model, tokenizer, output_path, sbic_data, additional_data, output_dir, model_name
):
    ab2label = {"a": "Hate", "b": "Non-hate"}
    sequence = "hn"
    label2ab = {v: k for k, v in ab2label.items()}
    id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

    data = additional_data

    output_path = (
        output_dir
        + f"{model_name.replace('/','-')}_simpleprompt_additional_predictions_{ab2label['a']}_{ab2label['b']}.csv"
    )
    inference(
        data, model, tokenizer, output_path, model_name, ab2label, sequence, simple=True
    )

    data = sbic_data

    output_path = (
        output_dir
        + f"{model_name.replace('/','-')}_simpleprompt_predictions_{ab2label['a']}_{ab2label['b']}.csv"
    )
    inference(
        data, model, tokenizer, output_path, model_name, ab2label, sequence, simple=True
    )

    ab2label = {"a": "Non-hate", "b": "Hate"}
    sequence = "nh"
    label2ab = {v: k for k, v in ab2label.items()}
    id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

    data = additional_data

    output_path = (
        output_dir
        + f"{model_name.replace('/','-')}_simpleprompt_additional_predictions_{ab2label['a']}_{ab2label['b']}.csv"
    )
    inference(
        data, model, tokenizer, output_path, model_name, ab2label, sequence, simple=True
    )

    data = sbic_data

    output_path = (
        output_dir
        + f"{model_name.replace('/','-')}_simpleprompt_predictions_{ab2label['a']}_{ab2label['b']}.csv"
    )
    inference(
        data, model, tokenizer, output_path, model_name, ab2label, sequence, simple=True
    )


def infer_model_persona_true(
    model, tokenizer, additional_data, sbic_data, output_dir, model_name
):
    countries = [
        "Australia",
        "United States",
        "United Kingdom",
        "South Africa",
        "Singapore",
    ]
    for i in range(5):  # PROMPTS
        for country in countries:
            ab2label = {"a": "Hate", "b": "Non-hate"}
            sequence = "hn"
            label2ab = {v: k for k, v in ab2label.items()}
            id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

            data = additional_data
            output_path = (
                output_dir
                + f"{model_name.replace('/','-')}_{country.replace(' ','_')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
            )
            inference(
                data,
                model,
                tokenizer,
                output_path,
                model_name,
                ab2label,
                sequence,
                definition=True,
                prompt_num=i,
                persona=True,
                country=country,
            )

            data = sbic_data
            output_path = (
                output_dir
                + f"{model_name.replace('/','-')}_{country.replace(' ','_')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
            )
            inference(
                data,
                model,
                tokenizer,
                output_path,
                model_name,
                ab2label,
                sequence,
                definition=True,
                prompt_num=i,
                persona=True,
                country=country,
            )

            ab2label = {"a": "Non-hate", "b": "Hate"}
            sequence = "nh"
            label2ab = {v: k for k, v in ab2label.items()}
            id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

            data = additional_data
            output_path = (
                output_dir
                + f"{model_name.replace('/','-')}_{country.replace(' ','_')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
            )
            inference(
                data,
                model,
                tokenizer,
                output_path,
                model_name,
                ab2label,
                sequence,
                definition=True,
                prompt_num=i,
                persona=True,
                country=country,
            )

            data = sbic_data
            output_path = (
                output_dir
                + f"{model_name.replace('/','-')}_{country.replace(' ','_')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
            )
            inference(
                data,
                model,
                tokenizer,
                output_path,
                model_name,
                ab2label,
                sequence,
                definition=True,
                prompt_num=i,
                persona=True,
                country=country,
            )


def infer_model_persona_false(
    model, tokenizer, sbic_data, additional_data, output_dir, model_name
):
    for i in range(5):  # PROMPTS
        ab2label = {"a": "Hate", "b": "Non-hate"}
        sequence = "hn"
        label2ab = {v: k for k, v in ab2label.items()}
        id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

        data = additional_data
        output_path = (
            output_dir
            + f"{model_name.replace('/','-')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
        )
        inference(
            data,
            model,
            tokenizer,
            output_path,
            model_name,
            ab2label,
            sequence,
            definition=True,
            prompt_num=i,
        )

        data = sbic_data
        output_path = (
            output_dir
            + f"{model_name.replace('/','-')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
        )
        inference(
            data,
            model,
            tokenizer,
            output_path,
            model_name,
            ab2label,
            sequence,
            definition=True,
            prompt_num=i,
        )

        ab2label = {"a": "Non-hate", "b": "Hate"}
        sequence = "nh"
        label2ab = {v: k for k, v in ab2label.items()}
        id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

        data = additional_data
        output_path = (
            output_dir
            + f"{model_name.replace('/','-')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
        )
        inference(
            data,
            model,
            tokenizer,
            output_path,
            model_name,
            ab2label,
            sequence,
            definition=True,
            prompt_num=i,
        )

        data = sbic_data
        output_path = (
            output_dir
            + f"{model_name.replace('/','-')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv"
        )
        inference(
            data,
            model,
            tokenizer,
            output_path,
            model_name,
            ab2label,
            sequence,
            definition=True,
            prompt_num=i,
        )


def run(
    simple: bool = False,
    persona: bool = False,
    output_dir: str = data_dir + "crehate_result/",
    model_name: str = "",
    custom_model_path=None,
    custom_model_tokenizer=None,
):

    os.makedirs(output_dir, exist_ok=True)
    sbic_data = pd.read_csv(data_dir + "CREHate_SBIC.csv", index_col=False)
    additional_data = pd.read_csv(data_dir + "CREHate_CP.csv", index_col=False)

    model, tokenizer = load_model(
        model_name=model_name,
        custom_model_path=custom_model_path,
        custom_model_tokenizer=custom_model_tokenizer,
    )

    if simple:
        infer_model_simple(
            model, tokenizer, sbic_data, additional_data, output_dir, model_name
        )
    elif persona:
        infer_model_persona_true(
            model, tokenizer, sbic_data, additional_data, output_dir, model_name
        )
    else:
        infer_model_persona_false(
            model, tokenizer, sbic_data, additional_data, output_dir, model_name
        )


if __name__ == "__main__":
    run(model_name="gpt-4-1106-preview")
