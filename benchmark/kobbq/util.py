import os, sys, re
import json
import time
import requests
import torch
from tqdm.auto import tqdm

from openai import OpenAI
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import anthropic

CLAUDE_MODEL = ["claude-instant-1.2", "claude-2.0"]
CLAUDE_API_KEY = os.environ.get("CLAUDE")
HYPERCLOVA_MODEL = {"clova-x": os.environ.get("CLOVA_URL")}
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Authorization": f'Bearer {os.environ.get("CLOVA")}',
}
KOALPACA_MODEL = ["KoAlpaca-Polyglot-12.8B"]
KOALPACA_MODEL_PATH = {"KoAlpaca-Polyglot-12.8B": "beomi/KoAlpaca-Polyglot-12.8B"}
GPT_MODEL = ["davinci", "gpt-3.5-turbo", "gpt-4"]
CUSTOM_MODEL_PREFIX = "custom"


def get_claude_response(prompt, model_name, max_tokens=128, max_try=10):
    assert model_name in CLAUDE_MODEL

    n_try = 0
    while True:
        if n_try == max_try:
            raise Exception("Something Wrong")

        try:
            time.sleep(1)
            c = anthropic.Client(CLAUDE_API_KEY)
            resp = c.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=model_name,
                max_tokens_to_sample=max_tokens,
            )
            break

        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupt")
        except Exception as e:
            print(e)
            print("Exception: Sleep for 5 sec")
            time.sleep(5)
            n_try += 1
            continue

    return resp["completion"]


def get_hyperclova_response(
    text,
    model_name,
    greedy,
    temperature=None,
    top_p=None,
    max_tokens=128,
    repeat_penalty=3,
    max_try=10,
):
    assert model_name in HYPERCLOVA_MODEL

    data = {
        "text_batch": text,
        "greedy": greedy,
        "max_tokens": max_tokens,
        "recompute": False,
        "repeat_penalty": repeat_penalty,
    }
    if not greedy:
        data["temperature"] = temperature
        data["top_p"]: top_p

    n_try = 0
    while True:
        if n_try > max_try:
            raise Exception("Something Wrong")

        try:
            response = requests.post(
                f"{HYPERCLOVA_MODEL[model_name]}",
                headers=HEADERS,
                data=json.dumps(data),
                timeout=60,
            )

            if response.status_code != 200:
                print(f"Error from internal API: {response.status_code}")
                time.sleep(5)
                n_try += 1
                continue

            outputs = response.json()["results"]
            results = [
                output["text"].strip().replace(prompt, "")
                for output, prompt in zip(outputs, data["text_batch"])
            ]
            break

        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupt")
        except Exception as e:
            print(e)
            print("Exception: Sleep for 5 sec")
            time.sleep(5)
            n_try += 1
            continue

    return results


def load_koalpaca(model_name="KoAlpaca-Polyglot-12.8B"):
    pipe = pipeline(
        "text-generation",
        torch_dtype=torch.bfloat16,
        model=KOALPACA_MODEL_PATH[model_name],
        tokenizer=KOALPACA_MODEL_PATH[model_name],
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"
    return pipe


def get_koalpaca_response(prompt, model_name, pipe, max_tokens, batch_size):
    assert model_name in KOALPACA_MODEL
    result = []
    try:
        for idx, out in enumerate(
            tqdm(
                pipe(prompt, batch_size=batch_size, max_new_tokens=max_tokens),
                total=len(prompt),
            )
        ):
            raw = out[0]["generated_text"]
            result.append(raw.split(prompt[idx])[-1])
    except Exception as e:
        print(e)
    return result


def check_gpt_input_list(history):
    check = True
    for i, u in enumerate(history):
        if not isinstance(u, dict):
            check = False
            break

        if not u.get("role") or not u.get("content"):
            check = False
            break

    return check


def get_gpt_response(
    text,
    model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=128,
    greedy=False,
    num_sequence=1,
    max_try=60,
    dialogue_history=None,
):
    assert model_name in GPT_MODEL
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
        if dialogue_history:
            if not check_gpt_input_list(dialogue_history):
                raise Exception(
                    "Input format is not compatible with chatgpt api! Please see https://platform.openai.com/docs/api-reference/chat"
                )
            messages = dialogue_history
        else:
            messages = []

        messages.append({"role": "user", "content": text})

        prompt = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0 if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence,
        }

    else:
        prompt = {
            "model": model_name,
            "prompt": text,
            "temperature": 0.0 if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence,
        }

    n_try = 0
    while True:
        if n_try == max_try:
            raise Exception("Something Wrong")

        try:
            if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
                time.sleep(1)
                res = client.chat.completions.create(**prompt)
                outputs = [o.message.content.strip("\n ") for o in res.choices]
            else:
                res = client.chat.completions.create(**prompt)
                outputs = [o.message.content.strip("\n ") for o in res.choices]
            break

        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupt")
        except Exception as e:
            print(res)
            print(e)
            print("Exception: Sleep for 5 sec")
            time.sleep(5)
            n_try += 1
            continue

    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs


def load_custom_model(model_path, model_tokenizer=""):
    if len(model_tokenizer) < 1:
        return pipeline("text-generation", model=model_path, device_map="auto")
    return pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_tokenizer,
        device_map="auto",
    )


def get_custom_model_response(prompt, model_name, pipe, max_tokens, batch_size):
    # assert CUSTOM_MODEL_PREFIX in model_name
    result = []
    try:
        for idx, out in enumerate(
            tqdm(
                pipe(prompt, batch_size=batch_size, max_new_tokens=max_tokens),
                total=len(prompt),
            )
        ):
            raw = out[0]["generated_text"]
            result.append(raw.split(prompt[idx])[-1])
    except Exception as e:
        print(e)
    return result
