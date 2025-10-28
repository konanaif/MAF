import json
import argparse
import random
import tqdm
import os
import logging
import sys
import tarfile
import time


import torch
import numpy as np


from nltk.corpus import stopwords

from functools import wraps
from datetime import datetime
from pathlib import Path
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from MAF.algorithms.postprocessing.casual_path_tracing.lib.nethook import (
    set_requires_grad,
)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


class LamatrexDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, reverse=False, *args, **kwargs):
        data_dir = Path(data_dir)
        known_loc = data_dir / "lama_trex.json"
        if not known_loc.exists():
            raise Exception

        with open(known_loc, "r") as f:
            self.data = json.load(f)
        if reverse:
            self.data = self.data[::-1]
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class function_time_saver:
    def __init__(self):
        self.function_timer = {}

    def save(self, func_name, val_dict):
        if func_name not in self.function_timer:
            self.function_timer[func_name] = {
                "total_time": 0,
                "cpu_time": 0,
                "gpu_time": 0,
                "gpu_mem_u": 0,
                "gpu_mem_r": 0,
                "func_file_path": val_dict["func_file_path"],
            }
        self.function_timer[func_name]["total_time"] += val_dict["total_time"]
        self.function_timer[func_name]["cpu_time"] += val_dict["cpu_time"]
        self.function_timer[func_name]["gpu_time"] += val_dict["gpu_time"]
        self.function_timer[func_name]["gpu_mem_u"] += val_dict["gpu_mem_u"]
        self.function_timer[func_name]["gpu_mem_r"] += val_dict["gpu_mem_r"]

    def logging(self, logger, header=None):
        keys = list(self.function_timer.keys())
        time_list = [self.function_timer[k]["total_time"] for k in keys]
        sorted_index = np.argsort(time_list)
        sorted_key = np.asarray(keys)[sorted_index]

        if header is not None:
            logger.info(header)
        for k in sorted_key:
            log_message = "\t[Function Log] '{}' => Total time: {:.4f}s (CPU time: {:.4f}s, GPU time: {:.4f}s) | GPU mem: (U) {:.2f}MB (R) {:.2f}MB\n\t\tFunction Path: '{}'".format(
                k,
                self.function_timer[k]["total_time"],
                self.function_timer[k]["cpu_time"],
                self.function_timer[k]["gpu_time"],
                self.function_timer[k]["gpu_mem_u"],
                self.function_timer[k]["gpu_mem_r"],
                self.function_timer[k]["func_file_path"],
            )
            logger.info(log_message)
        logger.info("==============================\n\n")


def set_utils(args):
    # Save Folder
    if "debug" not in args.save_root:
        if os.path.isdir(args.save_root) is True:
            print("Check your path!")
            import pdb

            pdb.set_trace()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    args.save_root += str(curr_time) + "_" + args.dataset_type
    if args.except_stopword:
        args.save_root += "_exStWd"
    os.makedirs(args.save_root, exist_ok=True)

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(args.save_root, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    func_time_saver = function_time_saver()
    # Timer
    if "debug" in args.save_root:
        current_dir = os.path.abspath(os.getcwd())
        imported_modules = {}
        for name, module in sys.modules.items():
            if module is None:
                continue
            if getattr(module, "__file__", "") is None:
                continue
            if module and getattr(module, "__file__", "").startswith(current_dir):
                imported_modules.update({name: module})
        for module_name, module in imported_modules.items():
            for name, func in vars(module).items():
                if (
                    callable(func)
                    and func.__module__ == module_name
                    and not getattr(func, "__is_decorated__", False)
                ):
                    if func.__name__ == "measure_time_and_memory":
                        continue
                    decorated_func = measure_time_and_memory(
                        logger=logger, func_time_saver=func_time_saver
                    )(
                        measure_time_and_memory(
                            logger=logger, func_time_saver=func_time_saver
                        )(func)
                    )
                    setattr(module, name, decorated_func)

    # Arg Backup
    with open(os.path.join(args.save_root, "args.txt"), "w") as f:
        json.dump(dict(vars(args)), f, indent=2)

    args.save_result_root = os.path.join(args.save_root, "results")
    args.save_inpinfo_root = os.path.join(args.save_root, "inp_info")
    os.makedirs(args.save_result_root, exist_ok=True)
    os.makedirs(args.save_inpinfo_root, exist_ok=True)

    return logger, func_time_saver


def measure_time_and_memory(logger=None, func_time_saver=None):
    def decorator(func):
        if getattr(func, "__is_decorated__", False):
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Measure CPU start time
            cpu_start_time = time.time()

            # Measure GPU start time (if CUDA is available)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
                gpu_start_time = torch.cuda.Event(enable_timing=True)
                gpu_end_time = torch.cuda.Event(enable_timing=True)
                gpu_start_time.record()
                gpu_start_abs_time = time.time()

                initial_memory = torch.cuda.memory_allocated()
                initial_reserved = torch.cuda.memory_reserved()

            result = func(*args, **kwargs)

            # Measure CPU end time
            cpu_end_time = time.time()
            cpu_elapsed_time = cpu_end_time - cpu_start_time

            # Measure GPU end time (if CUDA is available)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
                gpu_end_time.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded
                gpu_end_abs_time = time.time()
                gpu_elapsed_time = (
                    gpu_start_time.elapsed_time(gpu_end_time) / 1000.0
                )  # Convert milliseconds to seconds
                final_memory = torch.cuda.memory_allocated()
                final_reserved = torch.cuda.memory_reserved()
                memory_usage = (final_memory - initial_memory) / (1024 * 1024)
                memory_reserved = (final_reserved - initial_reserved) / (1024 * 1024)
            else:
                gpu_elapsed_time = None

            if gpu_start_abs_time is not None:
                total_start = min(cpu_start_time, gpu_start_abs_time)
                total_end = max(cpu_end_time, gpu_end_abs_time)
            else:
                total_start = cpu_start_time
                total_end = cpu_end_time
                gpu_elapsed_time = 0.0
                memory_usage = 0.0
                memory_reserved = 0.0
            total_elapsed_time = total_end - total_start

            # Get the absolute path of the function's module
            func_module = sys.modules[func.__module__]
            func_file_path = os.path.abspath(func_module.__file__)

            val_dict = {
                "total_time": total_elapsed_time,
                "cpu_time": cpu_elapsed_time,
                "gpu_time": gpu_elapsed_time,
                "gpu_mem_u": memory_usage,
                "gpu_mem_r": memory_reserved,
                "func_file_path": func_file_path,
            }
            func_time_saver.save(func.__name__, val_dict)
            # if logger:
            #     logger.info(log_message)
            # else:
            #     print(log_message)

            return result

        wrapper.__is_decorated__ = True
        return wrapper

    return decorator


def predict_from_normal_and_noise_input(
    prompt,
    flow_tracer,
    y,
    logger,
    mt,
    n_lev,
    num_noise_sample=3,
    num_normal_sample=3,
    noise_type="other",
    end_symbol=[".", "?"],
    out_num=1,
    correct_check_only=False,
):

    # num_noise_sample and num_normal_sample should be same.
    # their outputs are slightly different (it maybe the batch-wise operation in hardware-level.)

    normal_inp = make_inputs(mt.tokenizer, [prompt] * (num_normal_sample))
    # noise_inp = make_inputs(mt.tokenizer, [prompt] * (num_noise_sample))

    if out_num == 1:
        answer_t = flow_tracer.trace_normal(mt.model, normal_inp)
        answer = decode_tokens(mt.tokenizer, answer_t)
        if correct_check_only:
            return y in answer[0]
        if y not in answer[0]:
            return -1, -1, -1, -1, -1
        noise_rand_seed = 0
        while 1:
            corrupted_answer_t, used_token = flow_tracer.trace_corrupted(
                mt.model,
                mt.tokenizer,
                prompt,
                noise_level=n_lev,
                rand_seed=noise_rand_seed,
                noise_type=noise_type,
                num_noise_sample=num_noise_sample,
            )

            if answer_t.item() != corrupted_answer_t.item():
                break
            noise_rand_seed += 1
            logging.info(
                "[Noise Finder] Changing the seed to find noise that changes the output... Current Seed:{}, Ans:{}".format(
                    noise_rand_seed, answer[0]
                )
            )

    curr_total_token_num = normal_inp["input_ids"].shape[-1]

    if hasattr(mt.model, "transformer"):
        curr_total_block_num = len(mt.model.transformer.h)
    elif hasattr(mt.model, "gpt_neox"):
        curr_total_block_num = len(mt.model.gpt_neox.layers)

    return flow_tracer, answer, curr_total_token_num, curr_total_block_num, normal_inp


def make_inputs(tokenizer, prompts, device="cuda", pass_encoding=False):
    if pass_encoding is False:
        token_lists = [tokenizer.encode(p) for p in prompts]
    else:
        token_lists = prompts
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def get_stopwords(args, mt):
    stopword_list = stopwords.words("english")

    stopword_ids = []
    for stopword in stopword_list:
        token_ids = mt.tokenizer.encode(" " + stopword, add_special_tokens=False)
        if len(token_ids) == 1:
            stopword_ids.append(token_ids[0])
        token_ids = mt.tokenizer.encode(stopword, add_special_tokens=False)
        if len(token_ids) == 1:
            stopword_ids.append(token_ids[0])

    args.stwd_ids = sorted(list(set(stopword_ids)))


def predict_from_input(
    model,
    inp,
    multipred=False,
    end_symbol=[],
    mt=None,
    force_idx=None,
    use_mean=False,
    stwd_mask=None,
):
    # multipred option makes outputs until resulting one sentence output
    # Don't reduce the batch size...: https://discuss.pytorch.org/t/why-is-the-output-of-a-linear-layer-different-when-the-batch-size-is-1/93515
    if multipred is False:
        out = model(**inp)["logits"]
        if use_mean:
            probs = torch.softmax(out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
        else:
            probs = torch.softmax(out[:, -1], dim=1)

        if stwd_mask is not None:
            if force_idx is None:
                desc_idx = torch.argsort(probs, dim=1, descending=True)
                sorted_stwd_mask = stwd_mask[desc_idx]
                preds = desc_idx[sorted_stwd_mask][0]
                p = probs[:, preds][0]
            else:
                import pdb

                pdb.set_trace()
        else:
            if force_idx is None:
                p, preds = torch.max(probs, dim=1)
            else:
                p = probs[:, force_idx]
                preds = force_idx.repeat(p.shape[0])
        return preds.unsqueeze(0), p.unsqueeze(0), out
    else:
        import pdb

        pdb.set_trace()


def exclude_subsets(candidate, exclude_target):
    if len(exclude_target) == 0:
        return candidate
    new_candidate = []
    for cand in candidate:
        exclude_flag = False
        for ex in exclude_target:
            # ex is always smaller than cand, if the search proceeds with small number of steps.
            if len(set(ex).union(cand)) == len(cand):
                exclude_flag = True
                break
        if exclude_flag is False:
            new_candidate.append(cand)
    return new_candidate
