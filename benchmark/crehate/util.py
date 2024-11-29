import csv, os
from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    pipeline,
)
import time
import torch
from MAF.benchmark.crehate.process_data import (
    raw2prediction,
    prediction_2_label,
    make_prompt,
    check_gpt_input_list,
)


def load_model(
    model_name: str = "", custom_model_path=None, custom_model_tokenizer=None
):
    cache_dir = f'.cache/{model_name.replace("/", "-")}'
    if (custom_model_path != None) and (custom_model_tokenizer != None):
        tokenizer = AutoTokenizer.from_pretrained(
            custom_model_tokenizer, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            custom_model_path,
            device_map="auto",
            resume_download=True,
            cache_dir=cache_dir,
        )
        return model, tokenizer

    if "flan-t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", resume_download=True, cache_dir=cache_dir
        )
        return model, tokenizer

    if "llama" in model_name or "LLaMa" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name, use_fast=False, token=HUGGINGFACE_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            resume_download=True,
            cache_dir=cache_dir,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        return model, tokenizer

    if "gpt" in model_name or "claude" in model_name:
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", resume_download=True, cache_dir=cache_dir
    )
    return model, tokenizer


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
    # assert model_name in GPT_MODEL
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if (
        model_name.startswith("gpt-3.5-turbo") and "instruct" not in model_name
    ) or model_name.startswith("gpt-4"):
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
            outputs = ["something wrong"]
            break

        try:
            if (
                model_name.startswith("gpt-3.5-turbo") and "instruct" not in model_name
            ) or model_name.startswith("gpt-4"):
                time.sleep(0.5)
                res = client.chat.completions.create(**prompt)
                outputs = [o.message.content.strip("\n ") for o in res.choices]
            else:
                res = client.chat.completions.create(**prompt)
                outputs = [o.message.content.strip("\n ") for o in res.choices]
            break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except Exception as E:
            print(E)
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            n_try += 1
            continue

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def inference_on_single_data(
    context: str,
    model,
    tokenizer,
    model_name,
    ab2label,
    sequence,
    persona=False,
    country=None,
    simple=False,
    definition=True,
    prompt_num=None,
):
    num2label = ["Non-hate", "Hate"]

    hit_us, hit_uk, hit_au, hit_sa, hit_sg = 0, 0, 0, 0, 0
    evaluated_num = 0
    ooc = 0

    if persona:
        prompt = make_prompt(
            context,
            ab2label,
            persona=persona,
            country=country,
            definition=definition,
            prompt_num=prompt_num,
        )
    elif simple:
        prompt = make_prompt(context, ab2label, simple=simple)
    else:
        prompt = make_prompt(
            context, ab2label, definition=definition, prompt_num=prompt_num
        )

    if model_name.startswith("gpt"):
        raw = get_gpt_response(prompt, model_name)
        prediction = raw2prediction(raw, sequence)
        print("***GPT model result")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**input_ids, max_new_tokens=30)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw = result.replace(prompt, "")

    prediction = raw2prediction(raw, sequence)
    print("***RAW", raw)
    print("***Prediction", prediction)
    label = prediction_2_label(prediction, ab2label)
    return prediction


def inference(
    data,
    model,
    tokenizer,
    output_path,
    model_name,
    ab2label,
    sequence,
    persona=False,
    country=None,
    simple=False,
    definition=True,
    prompt_num=None,
):
    post_col = "Text"
    num2label = ["Non-hate", "Hate"]

    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "post", "US", "AU", "GB", "ZA", "SG", "prediction", "raw"]
        )
    done_id = pd.read_csv(output_path, encoding="utf-8")["id"].to_list()

    total_num = len(data)

    hit_us, hit_uk, hit_au, hit_sa, hit_sg = 0, 0, 0, 0, 0
    evaluated_num = 0
    ooc = 0

    tqdm_label = f"{model_name}-{prompt_num}"
    if persona:
        tqdm_label += f"-{country}"

    for idx, instance in tqdm(data.iterrows(), total=total_num, desc=tqdm_label):
        if instance["ID"] in done_id:
            continue
        evaluated_num += 1

        if persona:
            prompt = make_prompt(
                instance[post_col],
                ab2label,
                persona=persona,
                country=country,
                definition=definition,
                prompt_num=prompt_num,
            )
        elif simple:
            prompt = make_prompt(instance[post_col], ab2label, simple=simple)
        else:
            prompt = make_prompt(
                instance[post_col],
                ab2label,
                definition=definition,
                prompt_num=prompt_num,
            )
        print(prompt)

        if model_name.startswith("gpt"):
            result = get_gpt_response(prompt, model_name)
            raw = result
            prediction = raw2prediction(result, sequence)
            print(raw)
            print(prediction)
            label = prediction_2_label(prediction, ab2label)

        else:
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(**input_ids, max_new_tokens=30)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            raw = result.replace(prompt, "")
            prediction = raw2prediction(raw, sequence)
            print("raw:")
            print(raw)
            print("pred:")
            print(prediction)
            label = prediction_2_label(prediction, ab2label)

        if label not in ab2label.values():
            ooc += 1
            print("# ooc =", ooc)

        if label == num2label[int(float(instance["United_States_Hate"]))]:
            hit_us += 1
        if label == num2label[int(float(instance["United_Kingdom_Hate"]))]:
            hit_uk += 1
        if label == num2label[int(float(instance["Australia_Hate"]))]:
            hit_au += 1
        if label == num2label[int(float(instance["South_Africa_Hate"]))]:
            hit_sa += 1
        if label == num2label[int(float(instance["Singapore_Hate"]))]:
            hit_sg += 1

        open_trial = 0
        while True:
            if open_trial > 10:
                raise Exception("something wrong")

            try:
                with open(output_path, "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            instance["ID"],
                            instance[post_col],
                            num2label[int(float(instance["United_States_Hate"]))],
                            num2label[int(float(instance["Australia_Hate"]))],
                            num2label[int(float(instance["United_Kingdom_Hate"]))],
                            num2label[int(float(instance["South_Africa_Hate"]))],
                            num2label[int(float(instance["Singapore_Hate"]))],
                            label,
                            raw,
                        ]
                    )
                break
            except:
                print("open failed")
                continue
        print(
            f"[{model_name}]\tUS: {hit_us / evaluated_num:.4f}\tAU: {hit_au / evaluated_num:.4f}\tUK: {hit_uk / evaluated_num:.4f}\tSA: {hit_sa / evaluated_num:.4f}\tSG: {hit_sg / evaluated_num:.4f}"
        )
