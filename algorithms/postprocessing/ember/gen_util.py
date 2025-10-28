from tqdm import tqdm
from openai import OpenAI
from openai import (
    RateLimitError,
    APIError,
    Timeout,
    BadRequestError,
    APIConnectionError,
    InternalServerError,
)
from contextlib import contextmanager
import threading
import _thread
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import openai
import backoff
import concurrent.futures
import random
import os


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=""):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        timer.cancel()


def gpt4_answer(inputs_with_prompts, engine, max_tokens):
    outputs = []
    model = ParallelGPT(model_id=engine)
    for chunk_index, new_input in enumerate(tqdm(list(chunks(inputs_with_prompts, 5)))):
        generations = model.generate(
            new_input, max_new_tokens=max_tokens, temperature=0, num_return_sequences=1
        )["responses"]
        for generation in generations:
            outputs.append(generation[0])
    return outputs


def greedy_decoding_llama(pipeline, inputs_with_prompts, max_tokens):
    outputs = []
    for chunk_index, new_input in enumerate(list(chunks(inputs_with_prompts, 20))):
        sequence = pipeline(
            new_input,
            do_sample=False,
            num_return_sequences=1,
            add_special_tokens=True,
            max_new_tokens=max_tokens,
            temperature=None,
            top_p=None,
        )
        for idx, seq in enumerate(sequence):
            outputs.append(seq[0]["generated_text"][len(new_input[idx]) :])
        print("{} instances done".format((chunk_index + 1) * 20))

    return outputs


def greedy_decoding_mistral(inputs_with_prompts, engine, max_tokens):
    outputs = []
    messages = [{"role": "user"}]
    chatbot = transformers.pipeline(
        "text-generation", model=engine, max_new_tokens=max_tokens
    )
    for new_input in inputs_with_prompts:
        messages = [{"role": "user"}]
        messages[0]["content"] = new_input
        output = chatbot(messages)
        outputs.append(output)

    return outputs


class ParallelGPT:
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
            Timeout,
            BadRequestError,
            APIConnectionError,
            InternalServerError,
        ),
        max_tries=5,
        max_time=60,
    )
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def generate(
        self, text, max_new_tokens=1024, temperature=0, num_return_sequences=1, **kwargs
    ):
        if isinstance(text, str):
            text = [text]

        def process_text(t, idx):
            completion = self.completion_with_backoff(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": t,
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=0,
                n=num_return_sequences,
                **kwargs
            )
            return (completion, idx)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_text, t, idx) for idx, t in enumerate(text)
            ]
            completions = []
            for future in concurrent.futures.as_completed(futures):
                completions.append(future.result())

        completions_sorted = sorted(completions, key=lambda x: x[1])
        responses = [
            [
                completion[0].choices[i].message.content
                for i in range(num_return_sequences)
            ]
            for completion in completions_sorted
        ]
        completions = [completion[0] for completion in completions_sorted]

        return {"responses": responses, "completions": completions}
