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

from MAF.benchmark.crehate.util import (
    inference,
    load_model,
    inference_on_single_data,
)


def infer_model_simple(model, tokenizer, context: str, model_name):
    ab2label = {"a": "Hate", "b": "Non-hate"}
    sequence = "hn"
    label2ab = {v: k for k, v in ab2label.items()}
    id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

    result = {
        "hn": inference_on_single_data(
            context,
            model,
            tokenizer,
            model_name,
            {"a": "Hate", "b": "Non-hate"},
            sequence,
            simple=True,
        ),
        "nh": inference_on_single_data(
            context,
            model,
            tokenizer,
            model_name,
            {"a": "Non-hate", "b": "Hate"},
            sequence,
            simple=True,
        ),
    }
    return result


def infer_model_persona_true(model, tokenizer, context: str, model_name):
    countries = [
        "Australia",
        "United States",
        "United Kingdom",
        "South Africa",
        "Singapore",
    ]
    results = {
        f"prmpt_{i}": {c.replace(" ", ""): {} for c in countries} for i in range(5)
    }
    for i in range(5):  # PROMPTS
        for country in countries:
            ab2label = {"a": "Hate", "b": "Non-hate"}
            sequence = "hn"
            label2ab = {v: k for k, v in ab2label.items()}
            id2ab = {1: label2ab["Hate"], 0: label2ab["Non-hate"]}

            result = {
                "hn": inference_on_single_data(
                    context,
                    model,
                    tokenizer,
                    model_name,
                    {"a": "Hate", "b": "Non-hate"},
                    "hn",
                    definition=True,
                    prompt_num=i,
                    persona=True,
                    country=country,
                ),
                "nh": inference_on_single_data(
                    context,
                    model,
                    tokenizer,
                    model_name,
                    {"a": "Non-hate", "b": "Hate"},
                    "nh",
                    definition=True,
                    prompt_num=i,
                    persona=True,
                    country=country,
                ),
            }
            results[f"prmpt_{i}"][country.replace(" ", "")] = result
    return results


def infer_model_persona_false(model, tokenizer, context: str, model_name):
    results = {f"prmpt_{i}": {} for i in range(5)}
    for i in range(5):  # PROMPTS
        result = {
            "hn": inference_on_single_data(
                context,
                model,
                tokenizer,
                model_name,
                {"a": "Hate", "b": "Non-hate"},
                "hn",
                simple=True,
            ),
            "nh": inference_on_single_data(
                context,
                model,
                tokenizer,
                model_name,
                {"a": "Non-hate", "b": "Hate"},
                "nh",
                simple=True,
            ),
        }
        results[f"prmpt_{i}"] = result
    return results


def check_hatespeech(
    simple: bool = False,
    persona: bool = False,
    context: str = "",
    model_name: str = "gpt-4-1106-preview",
    custom_model_path=None,
    custom_model_tokenizer=None,
):

    model, tokenizer = load_model(
        model_name=model_name,
        custom_model_path=custom_model_path,
        custom_model_tokenizer=custom_model_tokenizer,
    )
    if simple:
        return infer_model_simple(model, tokenizer, context, model_name)
    elif persona:
        return infer_model_persona_true(model, tokenizer, context, model_name)
    else:
        return infer_model_persona_false(model, tokenizer, context, model_name)


if __name__ == "__main__":
    res = run(
        model_name="gpt-4-1106-preview",
        context="The National Authoritarian Terrorist Organization is behind this instability, just like they are responsible for all the other coos to disrupt opposing countries around the world!😔",
        simple=False,
        persona=True,
    )
    print(res)
