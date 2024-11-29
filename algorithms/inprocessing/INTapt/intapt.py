import os, sys
import argparse
import pickle
import json
from tqdm import tqdm
import glob
from typing import Tuple, Union, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from evaluate import load

from MAF.utils.common import fix_seed
from MAF.datamodule.intapt_datacollator import DataCollatorCTCWithPaddingCoraal
import MAF.algorithms.inprocessing.INTapt.util as util


class Mine(nn.Module):
    def __init__(self, input_size):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input_1, input_2):
        output = F.relu(self.fc1(torch.cat((input_1, input_2), axis=1)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class PromptGeneratorAttention(nn.Module):
    def __init__(self, args, embed_dim, num_heads, dropout, bias=True, do_train=False):
        super(PromptGeneratorAttention, self).__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.training = do_train

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, hidden_states):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class PromptGeneratorFeedForward(nn.Module):
    def __init__(
        self, args, hidden_size, activation_dropout, hidden_dropout, intermediate_size
    ):
        super(PromptGeneratorFeedForward, self).__init__()
        self.intermediate_dropout = torch.nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class PromptGenerator(nn.Module):
    def __init__(self, args, config):
        super(PromptGenerator, self).__init__()
        self.attention = PromptGeneratorAttention(
            args,
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = PromptGeneratorFeedForward(
            args,
            config.hidden_size,
            config.activation_dropout,
            config.hidden_dropout,
            config.intermediate_size,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.prompt_length = args.prompt_length

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states)
        )

        outputs = hidden_states
        if output_attentions:
            outputs += (attn_weights,)

        return outputs[:, : self.prompt_length, :]


class AccentClassifier(nn.Module):
    def __init__(self, args, config, num_labels):
        super(AccentClassifier, self).__init__()

        self.fc1 = nn.Linear(config.hidden_size, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 256)

        self.output_layer = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(args.accent_classifier_dropout)
        self.relu = nn.ReLU()

    def forward(self, input_feature):
        hidden_feature = self.relu(self.fc1(input_feature))
        hidden_feature = self.dropout(hidden_feature)
        hidden_feature = self.relu(self.fc2(hidden_feature))
        hidden_feature = self.dropout(hidden_feature)
        accent_feature = self.relu(self.fc3(hidden_feature))
        output_feauture = self.dropout(accent_feature)
        logits = self.output_layer(output_feauture)
        return logits, accent_feature


class AccentRegressor(nn.Module):
    def __init__(self, args):
        super(AccentRegressor, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(args.accent_classifier_dropout)
        self.relu = nn.ReLU()

    def forward(self, accent_feature):
        hidden_feature = self.relu(self.fc1(accent_feature))
        hidden_feature = self.dropout(hidden_feature)
        hidden_feature = self.relu(self.fc2(hidden_feature))
        hidden_feature = self.dropout(hidden_feature)
        output = self.relu(self.fc3(hidden_feature))
        return output


class AccentModule(nn.Module):
    def __init__(self, args, config, num_labels=6):
        super(AccentModule, self).__init__()
        self.accent_classifier = AccentClassifier(args, config, num_labels)
        self.accent_regressor = AccentRegressor(args)
        self.lamda = args.accent_lamda

    def forward(self, input_feature, asr_loss, batch):
        logits, accent_feature = self.accent_classifier(input_feature)
        accent_intensity = self.accent_regressor(accent_feature)
        return logits, accent_intensity, accent_feature


def load_model(
    args,
):
    model_path = glob.glob(
        args.hf_cache_dir + "/models--facebook--hubert-large-ls960-ft/" + "snapshots/*"
    )[0]
    prompt_generator_path = glob.glob(
        args.hf_cache_dir
        + "/models--esyoon--INTapt-HuBERT-large-coraal-prompt-generator/snapshots/*"
    )[0]
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    prompt_generator = PromptGenerator(args, model.config)
    prompt_generator.load_state_dict(
        torch.load(os.path.join(prompt_generator_path, "prompt_generator.pt"))
    )

    model.to(args.device)
    prompt_generator.to(args.device)
    return processor, model, prompt_generator


def cal_logits_w_prompt(args, model, batch, prompt):
    batch["feature"] = model.hubert.feature_extractor(batch["input_values"])
    batch["feature"] = model.hubert.feature_projection(batch["feature"].transpose(1, 2))
    model_input = torch.cat([prompt, batch["feature"]], dim=1)
    orig_hidden_states = model(
        batch["input_values"], return_dict="pt", output_hidden_states=True
    )["hidden_states"]
    pred_hidden_states_temp = model.hubert.encoder(
        model_input, return_dict="pt", output_hidden_states=True
    )
    last_hidden_state = pred_hidden_states_temp[0]
    pred_hidden_states = pred_hidden_states_temp[1]
    orig_hidden_state = orig_hidden_states[3]
    prompted_hidden_state = pred_hidden_states[3][:, args.prompt_length :, :]

    logits = model.dropout(last_hidden_state)
    logits = model.lm_head(logits[:, args.prompt_length :, :])

    return batch, orig_hidden_state, prompted_hidden_state, logits


def inference(args, model, prompt_generator, processor, metric, test_dataloader):
    model.eval()
    if args.eval_mode == "intapt":
        prompt_generator.eval()

    total_wer = 0.0
    steps = torch.tensor(len(test_dataloader)).to(args.device)

    for _, batch in enumerate(tqdm(test_dataloader)):
        batch = util.dict_to_device(batch, args.device)
        if args.eval_mode == "intapt":
            orig_pred = model(
                input_values=batch["input_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
            prompt = prompt_generator(orig_pred.hidden_states[3])
            _, _, _, prompt_logits = cal_logits_w_prompt(args, model, batch, prompt)
            wer = util.compute_metrics(
                prompt_logits, batch["labels"], processor, metric
            )
        elif args.eval_mode == "base":
            orig_pred = model(
                input_values=batch["input_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
            wer = util.compute_metrics(
                orig_pred.logits, batch["labels"], processor, metric
            )
    total_wer += torch.tensor(wer["wer"]).to(args.device)
    return total_wer / steps


parent_dir = os.environ["PYTHONPATH"]


class Arguments:
    def __init__(
        self,
        hf_cache_dir: str = os.environ["PYTHONPATH"] + "/MAF/data/INTapt/",
        batch_size: int = 2,
        dataset_name: str = "coraal",
        eval_metric: str = "wer",
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        prompt_length: int = 40,
        eval_mode: str = "intapt",
    ):
        self.hf_cache_dir = hf_cache_dir
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        self.device = device
        self.prompt_length = prompt_length
        self.eval_mode = eval_mode


def get_args():
    parser = argparse.ArgumentParser(description="CORAAL ASR test codes!")
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=os.environ["PYTHONPATH"] + "/MAF/data/INTapt/",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="coraal")
    parser.add_argument("--do_model_download", action="store_true")
    parser.add_argument("--eval_metric", type=str, default="wer")
    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--prompt_length", type=int, default=40)
    parser.add_argument("--eval_mode", type=str, default="intapt")
    return parser.parse_args()


def mitigate_intapt():
    args = Arguments()

    if args.eval_mode not in ["intapt", "base"]:
        print("Invalid eval mode. Choose from 'intapt' or 'base'...")
        quit()

    processor, model, prompt_generator = load_model(args)

    if args.eval_metric == "wer":
        metric = load("wer")
    elif args.eval_metric == "cer":
        metric = load("cer")

    data_collator = DataCollatorCTCWithPaddingCoraal(processor=processor, padding=True)

    test_dataset = load_dataset("esyoon/coraal_clean_test", cache_dir=args.hf_cache_dir)
    test_dataset = test_dataset["train"]

    test_speakers = ["ATL", "DCA", "DCB", "LES", "PRV", "ROC"]
    test_result_list = []

    for test_speaker in test_speakers:
        test_dataset_ = test_dataset.filter(lambda x: x["accent"] == test_speaker)
        test_dataloader = DataLoader(
            test_dataset_,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            pin_memory=True,
        )
        print("start testing for", test_speaker, "...")

        if args.eval_mode == "intapt":
            result = inference(
                args, model, prompt_generator, processor, metric, test_dataloader
            )
        elif args.eval_mode == "base":
            result = inference(args, model, None, processor, metric, test_dataloader)

        print(args.eval_metric, "for", test_speaker, ": ", result.item())
        test_result_list.append(result.item())

    test_avg_perf = sum(test_result_list) / len(test_result_list)
    perf_diff = max(test_result_list) - min(test_result_list)
    print("test avg performance: {:.4f} ".format(test_avg_perf))
    print("max - min performance: {:.4f}".format(perf_diff))

    result = {"test_avg_perf": test_avg_perf, "perf_diff": perf_diff}
    return result


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()
    fix_seed(1)
    mitigate_intapt()
