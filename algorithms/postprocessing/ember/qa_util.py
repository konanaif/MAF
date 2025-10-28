from tqdm import tqdm
from openai import OpenAI
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
from tqdm import tqdm
from MAF.algorithms.postprocessing.ember.gen_util import gpt4_answer


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


def prompt_generation(scoring, question, output, reference):
    qa_eval_prompt = """You are a helpful assistant tasked with evaluating the quality of an answer to a given question.
Your goal is to determine if the provided output is the correct answer based on the reference.

Do NOT provide any explanation for your choice.
The epistemic markers in the output may be deceiving.
You should respond using ONLY “Yes” or "No” without including any other words.

# Question:
{question}
# Reference:
{reference}
# Output:
{output}
# Is output correct? Your response should be either Yes” or No
"""

    # CoT version
    #     qa_eval_prompt="""You are a helpful assistant tasked with evaluating the quality of an answer to a given question.
    # Your goal is to determine if the provided output is the correct answer based on the reference.
    # You should first provide a brief explanation of your evaluation, and then always
    # end your response with either “Therefore, yes Output is correct.” or “Therefore, no Output is incorrect.” verbatim.
    # Do NOT say both / neither are good.
    # Do NOT output any other words.
    # Do NOT say “Yes, Output is correct” or “No, Output is incorrect” at the beginning. You should do reasoning and thinking **before** claiming whether it is correct or incorrect

    # # Question:
    # {question}
    # # Reference:
    # {reference}
    # # Output:
    # {output}
    # Is output correct?
    # # Decision (Give a brief explanation of your evaluation followed by either “Therefore, yes Output is correct.” or “Therefore, no Output is incorrect.” verbatim. Always claim which is better at the end.):
    # """

    return qa_eval_prompt.format(question=question, output=output, reference=reference)


def prepare_qa_inputs_rule(data, position, reader):
    markers = {}
    markers["str"] = {}
    markers["str"]["exp"] = [
        "I am confident with the answer.",
        "I am certain with the answer.",
        "I know the answer.",
        "Absolutely certain with the answer.",
        "I'm confident with the answer.",
        "Certainty level: high.",
        "High degree of certainty.",
        "High level of confidence.",
        "Undoubtedly, the answer is correct.",
        "Very confident with the answer.",
        "High degree of confidence.",
        "Confidence level: high.",
        "Completely certain.",
        "Definitely, the answer is correct.",
        "I can confidently say, the answer is correct.",
        "Very certain with the answer.",
        "Completely confident with the answer.",
        "My certainty level for this answer is high.",
        "Highly confident with the answer.",
        "My confidence level for this answer is high.",
    ]
    markers["str"]["pop"] = [
        4585,
        3833,
        2661,
        2215,
        1390,
        1110,
        1021,
        938,
        857,
        828,
        792,
        766,
        731,
        650,
        575,
        531,
        507,
        483,
        462,
        461,
    ]

    markers["weak"] = {}
    markers["weak"]["exp"] = [
        "I'm not sure with the answer.",
        "I cannot provide a definitive answer.",
        "It is possible.",
        "I cannot say for certain.",
        "The answer seems unlikely.",
        "Not completely certain with the answer.",
        "Not entirely certain with the answer.",
        "I don't know the answer.",
        "Not entirely clear with the answer.",
        "I'm not entirely sure with the answer.",
        "It could be correct.",
        "Not 100% certain with the answer.",
        "It is not clear.",
        "Cannot be completely certain with the answer.",
        "Not completely sure with the answer.",
        "The answer is not entirely accurate.",
        "I am unsure with the answer.",
        "I cannot say with absolute certainty.",
        "I cannot be certain with the answer.",
        "Not 100% sure with the answer.",
    ]
    markers["weak"]["pop"] = [
        2338,
        1931,
        1847,
        1795,
        1192,
        1114,
        947,
        804,
        762,
        748,
        737,
        723,
        675,
        626,
        606,
        582,
        549,
        531,
        343,
        336,
    ]
    str_exps = random.choices(
        population=markers["str"]["exp"], weights=markers["str"]["pop"], k=len(data)
    )
    weak_exps = random.choices(
        population=markers["weak"]["exp"], weights=markers["weak"]["pop"], k=len(data)
    )

    if position == "front":
        for idx, d in enumerate(data):
            d["answer_{}_str".format(reader)] = (
                str_exps[idx] + " Answer: " + d["answer_{}".format(reader)]
            )
            d["answer_{}_weak".format(reader)] = (
                weak_exps[idx] + " Answer: " + d["answer_{}".format(reader)]
            )
            d["answer_{}_plain".format(reader)] = d["answer_{}".format(reader)]

    elif position == "back":
        for idx, d in enumerate(data):
            d["answer_{}_str".format(reader)] = (
                "Answer: " + d["answer_{}".format(reader)] + " " + str_exps[idx]
            )
            d["answer_{}_weak".format(reader)] = (
                "Answer: " + d["answer_{}".format(reader)] + " " + weak_exps[idx]
            )
            d["answer_{}_plain".format(reader)] = d["answer_{}".format(reader)]
    else:
        raise TypeError


def prepare_qa_inputs_gpt(data, engine, reader):
    output_revision_prompt = """You are given an question along with the LLM's original output for that question. Your task is to revise the output by adding epistemic markers (words that convey certainty or uncertainty).

    Do not change the meaning of the original output. Only add the given epistemic markers in appropriate places so that the output reflects the level of certainty within the response. You must include the given epistemic markers in the revised output.

    Provide only the revised output and nothing else.

    Question: {question}

    Given Epistemic Marker: {em}

    Original output: {output}"""
    markers = {}
    markers["str"] = {}
    markers["str"]["exp"] = [
        "I am confident",
        "I am certain",
        "I know",
        "Absolutely certain",
        "I'm confident",
        "Certainty level: high",
        "High degree of certainty",
        "High level of confidence",
        "Undoubtedly",
        "Very confident",
        "High degree of confidence",
        "Confidence level: high",
        "Completely certain",
        "Definitely",
        "I can confidently say",
        "Very certain",
        "Completely confident",
        "My certainty level for this answer is high",
        "Highly confident",
        "My confidence level for this answer is high",
    ]
    markers["str"]["pop"] = [
        4585,
        3833,
        2661,
        2215,
        1390,
        1110,
        1021,
        938,
        857,
        828,
        792,
        766,
        731,
        650,
        575,
        531,
        507,
        483,
        462,
        461,
    ]

    markers["weak"] = {}
    markers["weak"]["exp"] = [
        "I'm not sure",
        "I cannot provide a definitive answer",
        "It is possible",
        "I cannot say for certain",
        "Seems unlikely",
        "Not completely certain",
        "Not entirely certain",
        "I don't know",
        "Not entirely clear",
        "I'm not entirely sure",
        "It could be",
        "Not 100% certain",
        "It is not clear",
        "Cannot be completely certain",
        "Not completely sure",
        "Not be entirely accurate",
        "I am unsure",
        "I cannot say with absolute certainty",
        "I cannot be certain",
        "Not 100% sure",
    ]
    markers["weak"]["pop"] = [
        2338,
        1931,
        1847,
        1795,
        1192,
        1114,
        947,
        804,
        762,
        748,
        737,
        723,
        675,
        626,
        606,
        582,
        549,
        531,
        343,
        336,
    ]

    str_exps = random.choices(
        population=markers["str"]["exp"], weights=markers["str"]["pop"], k=len(data)
    )
    weak_exps = random.choices(
        population=markers["weak"]["exp"], weights=markers["weak"]["pop"], k=len(data)
    )

    str_inputs1 = []
    weak_inputs1 = []

    for idx, d in enumerate(data):
        str_inputs1.append(
            output_revision_prompt.format(
                em=str_exps[idx],
                question=d["question"][0].upper() + d["question"][1:],
                output=d["answer_{}".format(reader)],
            )
        )
        weak_inputs1.append(
            output_revision_prompt.format(
                em=weak_exps[idx],
                question=d["question"][0].upper() + d["question"][1:],
                output=d["answer_{}".format(reader)],
            )
        )

    str_outputs1 = gpt4_answer(
        inputs_with_prompts=str_inputs1, engine=engine, max_tokens=300
    )
    weak_outputs1 = gpt4_answer(
        inputs_with_prompts=weak_inputs1, engine=engine, max_tokens=300
    )

    for idx, d in enumerate(data):
        d["answer_{}_str".format(reader)] = str_outputs1[idx]
        d["answer_{}_weak".format(reader)] = weak_outputs1[idx]
        d["answer_{}_plain".format(reader)] = d["answer_{}".format(reader)]
        d["str"] = str_exps[idx]
        d["weak"] = weak_exps[idx]


def integer_detector(output: str):
    integers = ["1", "2", "3", "4", "5", "0"]
    found = False
    for ints in integers:
        if ints in output:
            found = True
            return int(ints)
    if not found:
        return None


def yesno_detector(output: str):
    if "no" in output.lower():
        return 0
    elif "yes" in output.lower():
        return 1
    else:
        return None


# # CoT
# def yesno_detector (output: str):
#     if len(output.split())>100:
#         output=output[-50:]

#     if 'no' in output.lower() and 'incorrect' in output.lower():
#         return 0
#     elif 'yes' in output.lower() and 'correct' in output.lower():
#         return 1
#     else:
#         return None
