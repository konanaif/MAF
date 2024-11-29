import json
import scipy
import argparse
import re
import csv
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def preprocess(args, is_demo: bool = False):
    def change_unk(x: str, unk: str):
        if x == "알 수 없음":
            return unk
        else:
            return x

    def change_unk_choice(x: str, unk: str):
        return [change_unk(choice, unk) for choice in x]

    def permute(df: pd.DataFrame):
        processed_df = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)
        permute_choices = pd.DataFrame(
            np.concatenate(df["choices"].apply(lambda x: scipy.linalg.circulant(x))),
            columns=["A", "B", "C"],
        )
        processed_df = pd.concat([processed_df, permute_choices], axis=1)
        processed_df["sample_id"] = processed_df["sample_id"] + [
            "-0",
            "-1",
            "-2",
        ] * len(df)
        processed_df["choices"] = processed_df.apply(lambda x: [x.A, x.B, x.C], axis=1)
        return processed_df

    def make_prefix(df_prompts, prompt_id):
        prefix = df_prompts[df_prompts["prompt_id"] == prompt_id]["instruction"].item()
        return prefix + "\n\n"

    def make_prompt(df_prompts, prompt_id, context, question, A, B, C):
        prompt_row = df_prompts[df_prompts["prompt_id"] == prompt_id]

        prompt = prompt_row["context"].item() + context + "\n"
        prompt += prompt_row["question"].item() + question + "\n"
        prompt += prompt_row["a"].item() + A + "\n"
        prompt += prompt_row["b"].item() + B + "\n"
        prompt += prompt_row["c"].item() + C + "\n"
        prompt += prompt_row["answer"]

        return prompt

    def process(df, df_prompts, prompt_id):
        unk = df_prompts[df_prompts["prompt_id"] == prompt_id]["unknown"].item()

        df["choices"] = df["choices"].apply(lambda x: change_unk_choice(x, unk))
        df["answer"] = df["answer"].apply(lambda x: change_unk(x, unk))

        df = permute(df)
        df["answer_abc"] = df[["answer", "A", "B", "C"]].apply(
            lambda x: x[x == x.answer].index.tolist()[-1], axis=1
        )

        prefix = make_prefix(df_prompts, prompt_id)
        df["query"] = df.apply(
            lambda x: make_prompt(
                df_prompts, prompt_id, x.context, x.question, x.A, x.B, x.C
            ),
            axis=1,
        )

        df = df.sort_values(by=["sample_id"])
        return df, prefix

    def to_tsv(df, tsv_path):
        Path(tsv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tsv_path, sep="\t", index=False)

    def to_json(df, prefix, json_path):
        data = {
            "ver": "test",
            "prefix": prefix,
            "data": df[
                ["query", "A", "B", "C", "answer_abc", "sample_id"]
            ].values.tolist(),
        }

        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    if is_demo:
        df = pd.DataFrame(
            {
                "sample_id": [args.sample_id],
                "context": [args.context],
                "question": [args.question],
                "choices": [args.choices],
                "biased_answer": [args.biased_answer],
                "answer": [args.answer],
                "prediction": [""],
            }
        )
    else:
        df = pd.read_csv(args.samples_tsv_path, sep="\t")
        df["choices"] = df["choices"].map(eval)

    df_prompts = pd.read_csv(args.prompt_tsv_path, sep="\t")
    df, prefix = process(df, df_prompts, args.prompt_id)
    if is_demo:
        return {
            "ver": "test",
            "prefix": prefix,
            "data": df[
                ["query", "A", "B", "C", "answer_abc", "sample_id"]
            ].values.tolist(),
            "choices": df["choices"].tolist(),
        }

    if args.evaluation_json_path:
        to_json(df, prefix, args.evaluation_json_path)
    if args.evaluation_tsv_path:
        df = df.drop("query", axis=1)
        to_tsv(df, args.evaluation_tsv_path)


def raw2prediction(raw, choices):
    try:
        prediction = re.search("^\s*\(?(?P<raw>[^\.\n]*)\s*", raw).groupdict()["raw"]
    except:
        prediction = ""

    prediction = prediction.replace("없습니다", "없음").replace("입니다", "")

    prediction_upper = prediction.upper()

    if prediction_upper and (
        prediction_upper[0] in choices.keys()
    ):  # starts with A, B, C
        try:
            choice = (
                re.search("[:)]\s*(?P<choice>.*)\s*", prediction_upper)
                .groupdict()["choice"]
                .strip()
            )

            if len(choice) == 0:
                raise Exception

            if choices[prediction_upper[0]] == choice.lower():
                return prediction_upper[0]
            elif (
                sum(
                    prediction_upper.count(choice.upper())
                    for choice in choices.values()
                )
                > 1
            ):  # out-of-choice
                return prediction
            else:
                # print(f"'{prediction_upper[0]}' should be '{choices[prediction_upper[0]]}', but '{prediction}' found")
                return prediction_upper[0]

        except:
            if (
                sum(prediction_upper.count(alphabet) for alphabet in choices.keys())
                == 1
            ):
                prediction = prediction_upper[0]
                return prediction

    if prediction.lower() in choices.values():  # one of choices
        return list(choices.keys())[list(choices.values()).index(prediction.lower())]

    else:
        try:
            raw = re.search("\*\*['\"]?(?P<answer>[^\.\n\*'\"]*)\s*", raw).groupdict()[
                "answer"
            ]
            return raw2prediction(raw, choices)
        except:
            pass

        try:
            raw = re.search(
                "답변?[은:]\s*['\"]?(?P<answer>[^\.\n\*'\"]*)\s*", raw
            ).groupdict()["answer"]
            return raw2prediction(raw, choices)
        except:
            pass

        try:
            raw = re.search("['\"](?P<answer>[^\.\n\*'\"]*)\s*", raw).groupdict()[
                "answer"
            ]
            return raw2prediction(raw, choices)
        except:
            pass

        raw_upper = str(raw).upper()
        count = 0
        answer_alphabet = ""
        for alphabet, choice in choices.items():
            if len(re.findall(f"{alphabet}[:)]", raw_upper)) > 0:
                answer_alphabet = alphabet
                count += 1
        if count == 1:
            return answer_alphabet

        return raw


def postprocess_predictions(args):
    def postprocess(x_result, df_processed, ooc_path=None):
        x_processed = df_processed[df_processed["sample_id"] == x_result.guid]

        if len(x_processed) == 0:
            raise ValueError(f"{x_result.guid} Not Found")

        choices = {
            "A": x_processed.A.item().lower(),
            "B": x_processed.B.item().lower(),
            "C": x_processed.C.item().lower(),
        }

        prediction = raw2prediction(x_result.raw, choices)

        if prediction not in ["A", "B", "C"] and ooc_path:
            with open(ooc_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([x_result.guid, choices, x_result.raw, prediction])

        return prediction

    tqdm.pandas()

    if args.ooc_path:
        Path(args.ooc_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.ooc_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "choices", "raw", "processed"])

    df_result = pd.read_csv(args.predictions_tsv_path, delimiter="\t")  # 모델의 예측 데이터
    df_processed = pd.read_csv(args.preprocessed_tsv_path, delimiter="\t")  # 평가할 데이터

    df_result["prediction"] = df_result.progress_apply(
        lambda x: postprocess(x, df_processed, args.ooc_path), axis=1
    )
    df_result.to_csv(args.predictions_tsv_path, sep="\t", index=False)
    print("out-of-choice count", (~df_result["prediction"].isin(["A", "B", "C"])).sum())


def predictions_to_evaluation(args):
    def abc_2_prediction(x_processed, df_result):
        pred = df_result[df_result.guid == x_processed.sample_id].prediction

        if len(pred) == 1:
            pred = pred.item()
        elif len(pred) == 0:
            raise ValueError(f"{x_processed.sample_id} Not Found")
        else:
            raise ValueError(f"{len(pred)} {x_processed.sample_id} Found")

        if pred in ["A", "B", "C"]:
            pred = x_processed[pred]

        return pred

    def to_model_evaluation_tsv(df, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)

    tqdm.pandas()

    df_result = pd.read_csv(args.predictions_tsv_path, delimiter="\t")
    df_evaluation = pd.read_csv(args.preprocessed_tsv_path, delimiter="\t")

    df_evaluation["prediction"] = df_evaluation.progress_apply(
        lambda x: abc_2_prediction(x, df_result), axis=1
    )

    to_model_evaluation_tsv(df_evaluation, args.output_path)
